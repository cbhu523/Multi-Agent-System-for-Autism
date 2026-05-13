"""
ADOS-2 Module 4  --  ProCoT Baseline
======================================

Adaptation to ADOS-2 SLD trait discovery
------------------------------------------
ProCoT defines three prompting schemes (§3, Figure 1):

  (1) Standard Prompting    p(r | D, C)
      Given task background D and conversation history C, generate response r.
      No strategy awareness. Direct question generation.

  (2) Proactive Prompting   p(a, r | D, C, A)
      Given D, C, and action set A, select action a ∈ A then generate r.
      The LLM is aware of available strategies but does NOT reason about goal.

  (3) ProCoT Prompting      p(t, a, r | D, C, A)   ← proposed method
      First generate a THOUGHT t (descriptive reasoning about the current
      diagnostic state and what information is still missing), then select
      action a, then generate response r.
      This endows the LLM with goal-planning capability.

ADOS-2 mapping:
  D  = task background: ADOS-2 Module 4 context + SLD trait definitions
  C  = dialog_history (doctor-patient conversation so far)
  A  = STRATEGIES (6 clinical questioning strategies)
  t  = Thought: analysis of which SLD traits have been observed, which are
       uncertain, and what questioning angle would best surface new traits
  a  = Selected strategy ∈ STRATEGIES
  r  = Doctor's next question (generated conditioned on t and a)

Ablation modes
--------------
  "procot"      -- Full ProCoT: Thought → Strategy → Question  (proposed)
  "proactive"   -- Proactive only: Strategy → Question  (no Thought)
  "standard"    -- Standard: direct Question generation  (no strategy)
  "random"      -- Uniform random strategy selection (shared baseline)
  "round_robin" -- Fixed rotation (shared baseline)

All five modes are run on the same uid-scenes for direct comparison.

Key design invariants (shared with BED-LLM / GDP-Zero pipelines)
------------------------------------------------------------------
  GT is NEVER visible during any episode.
  Post-hoc evaluation uses GT for all metrics.
  Same infrastructure: SnippetBank, EmbeddingIndex, PatientAgent,
  DoctorAgent, TraitDetector, TraitBelief, compute_summary, etc.
"""

import os, json, random, time, math, re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime

import numpy as np
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from autogen import AssistantAgent

# =========================================================
# Constants  (identical across all baselines for comparability)
# =========================================================
SNIPPET_JSONL        = "./dataset/snippet_bank_with_probs.jsonl"
LOG_DIR              = "logs_procot"
MIN_SNIPPETS_PER_UID = 1
MAX_TURNS            = 20
FP_PENALTY_ALPHA     = 0.3
EARLY_STOP_THRESHOLD = 1.5   # same as BED-LLM for fair comparison

# Belief hyper-parameters
BELIEF_ALPHA0    = 1.5
BELIEF_BETA0     = 3.0
BELIEF_LR_PROXY  = 1.2
BELIEF_MAX_ALPHA = 12.0

PHENOMENA = ["1","2","3","4","5","6","7","8","9","10"]

TRAIT_ID_MAP = {
    "1":"Echoic Repetition",        "2":"Unconventional Content",
    "3":"Pronoun Displacement",     "4":"Incongruous Humor Timing",
    "5":"Formalistic Language Use", "6":"Superfluous Phrase Attachment",
    "7":"Excessive Social Phrasing","8":"Monotone Social Expression",
    "9":"Stereotyped Media Quoting","10":"Clichéd Verbal Substitutions",
}

TRAIT_WEIGHTS = {"1":2.0, "4":1.4, "5":1.4, "9":2.0}

STRATEGIES = [
    "Open-ended question",
    "Emotion-oriented question",
    "Hypothetical prompt",
    "Multi-step guidance",
    "Perspective-taking question",
    "Correction-inducing question",
]

STRATEGY_GUIDANCE = {
    "Open-ended question":
        "Ask a broad, open-ended question. Avoid yes/no. Use 'how', 'what', 'tell me about'.",
    "Emotion-oriented question":
        "Ask about feelings or emotional reactions. Empathetic, subjective focus.",
    "Hypothetical prompt":
        "Present a hypothetical scenario. Use 'imagine if', 'what would you do if'.",
    "Multi-step guidance":
        "Break topic into steps. Ask patient to walk through a process.",
    "Perspective-taking question":
        "Ask patient to consider another person's viewpoint.",
    "Correction-inducing question":
        "Make a gentle misstatement and invite the patient to correct it.",
}

EPS = 1e-9

# ProCoT task background (D in the paper) -- shown once per prompt
_TASK_BACKGROUND = """\
You are conducting an ADOS-2 Module 4 autism language assessment.
Your goal is to surface Social Language Disorder (SLD) traits in the patient's speech.
The 10 SLD trait categories are:
  F1  Echoic Repetition        -- verbatim mimicry of what was just said
  F2  Unconventional Content   -- peculiar or oddly chosen phrasing
  F3  Pronoun Displacement     -- substitutes 'you' for 'I', uses 3rd-person self
  F4  Incongruous Humor        -- inserts humour during serious discussion
  F5  Formalistic Language     -- overly formal or archaic register
  F6  Superfluous Phrases      -- redundant filler expressions attached to answers
  F7  Excessive Social Phrasing-- overuses social politeness formulas
  F8  Monotone Social Express. -- repeats flat social phrases without variation
  F9  Stereotyped Media Quoting-- quotes media stereotypically
  F10 Clichéd Substitutions    -- uses clichés instead of direct responses
"""

_STRATEGY_LIST = "\n".join(
    f"  {i+1}. {s} -- {STRATEGY_GUIDANCE[s]}"
    for i, s in enumerate(STRATEGIES))


# =========================================================
# Utilities  (identical across all baselines)
# =========================================================
def parse_uid(uid: str) -> Tuple[str, Optional[str]]:
    parts = uid.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0], parts[1]
    return uid, None


_JSON_FENCE_RE    = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)
_TRAIT_NUMBERS_RE = re.compile(r"\b(\d{1,2})\b")


def _sanitize(t: str) -> str:
    return t.lstrip("\ufeff\u200b\u200c\u200d").strip()


def _clean_text(text: str, max_chars: int = 2000) -> str:
    if not text:
        return ""
    cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', ' ', text)
    cleaned = re.sub(r'[\ufffd\ufffe\uffff]', ' ', cleaned)
    cleaned = re.sub(r'[ \t]{2,}', ' ', cleaned)
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars] + "…"
    return cleaned.strip()


def _try_parse_json(text: str) -> Optional[dict]:
    if not text: return None
    text = _sanitize(text)
    if text.startswith("{"):
        try: return json.loads(text)
        except: pass
    m = _JSON_FENCE_RE.search(text)
    if m:
        try: return json.loads(_sanitize(m.group(1)))
        except: pass
    s, e = text.find("{"), text.rfind("}")
    if s != -1 and e > s:
        c = _sanitize(text[s:e+1])
        try: return json.loads(c)
        except:
            fixed = re.sub(r",\s*([}\]])", r"\1", c)
            try: return json.loads(fixed)
            except: pass
    return None


def _extract_patient_reply_fallback(text: str) -> str:
    if not text: return ""
    text = text.strip()
    for pat in [r'"patient_reply"\s*:\s*"((?:[^"\\]|\\.)*)"',
                r"patient_reply\s*:\s*(.+?)(?:\n|$)"]:
        m = re.search(pat, text, re.IGNORECASE | re.DOTALL)
        if m: return m.group(1).strip().strip('"')
    for prefix in ["Patient:", "patient:", "Response:", "response:"]:
        if text.startswith(prefix): return text[len(prefix):].strip()
    return text


def _extract_trait_ids(text: str) -> List[str]:
    if not text: return []
    text = text.strip()
    if text.lower().replace(" ", "") in {
            "0","none","nil","null","no","notraits"}: return []
    seen, valid = set(), []
    for n in _TRAIT_NUMBERS_RE.findall(text):
        if n in PHENOMENA and n not in seen and n != "0":
            valid.append(n); seen.add(n)
    return valid


def llm_json_call(llm, system_prompt: str, user_prompt: str,
                  required_key: str = "patient_reply",
                  max_retries: int = 6,
                  fail_log_dir: str = "logs_procot/json_failures") -> dict:
    os.makedirs(fail_log_dir, exist_ok=True)
    last_text, last_err = "", None
    for attempt in range(max_retries):
        try:
            prefix = ("" if attempt == 0 else
                      f"STRICT JSON REQUIREMENT (attempt {attempt+1}/{max_retries}):\n"
                      f'Return ONLY a JSON object with key "{required_key}".\n'
                      "Start with {{ end with }}. No other text.\n\n")
            result = llm.invoke([HumanMessage(
                content=f"{system_prompt}\n\n{prefix}{user_prompt}")])
            last_text = (result.content if hasattr(result, "content")
                         else str(result)) or ""
            if not last_text.strip(): raise ValueError("Empty response")
            parsed = _try_parse_json(last_text)
            if parsed and required_key in parsed: return parsed
            if parsed:
                for alt in ["reply","response","text","output","answer"]:
                    if alt in parsed:
                        parsed[required_key] = parsed[alt]; return parsed
            raise ValueError(f"Missing key '{required_key}'")
        except Exception as e:
            last_err = e
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            try:
                with open(os.path.join(fail_log_dir,
                          f"fail_{ts}_a{attempt+1}.txt"), "w") as f:
                    f.write(f"ERROR: {e}\n\nOUTPUT:\n{last_text}\n")
            except: pass
            if attempt == 1:
                try:
                    llm = ChatOpenAI(
                        model=getattr(llm, "model_name", "gpt-4.1-nano"),
                        temperature=0.0)
                except: pass
            time.sleep(min(0.5 * (attempt + 1), 4.0))
    fb = _extract_patient_reply_fallback(last_text)
    if fb: return {required_key: fb, "_fallback": True}
    raise RuntimeError(
        f"llm_json_call failed after {max_retries} retries. Last: {last_err}.")


def clamp01(x: Any) -> float:
    try: return max(0.0, min(1.0, float(x)))
    except: return 0.0


def sigmoid(x: float) -> float:
    if x >= 0: return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x); return z / (1.0 + z)


def logit(p: float) -> float:
    p = min(1 - 1e-6, max(1e-6, p))
    return math.log(p / (1.0 - p))


# =========================================================
# TraitBelief
# =========================================================
@dataclass
class TraitBelief:
    alpha: Dict[str, float] = field(default_factory=dict)
    beta:  Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        for tid in PHENOMENA:
            if tid not in self.alpha: self.alpha[tid] = BELIEF_ALPHA0
            if tid not in self.beta:  self.beta[tid]  = BELIEF_BETA0

    def mean(self, tid: str) -> float:
        a, b = self.alpha[tid], self.beta[tid]
        return a / (a + b)

    def all_means(self) -> Dict[str, float]:
        return {tid: self.mean(tid) for tid in PHENOMENA}

    def uncertainty(self, tid: str) -> float:
        a, b = self.alpha[tid], self.beta[tid]
        conc = (a - 1.0) ** 2 + (b - 1.0) ** 2
        return 1.0 / (1.0 + 0.1 * conc)

    def total_uncertainty(self) -> float:
        return sum(self.uncertainty(tid) for tid in PHENOMENA)

    def uncertainty_dict(self) -> Dict[str, float]:
        return {tid: self.uncertainty(tid) for tid in PHENOMENA}

    def proxy_update(self, detected: List[str]) -> None:
        for tid in detected:
            self.alpha[tid] = min(BELIEF_MAX_ALPHA,
                                  self.alpha[tid] + BELIEF_LR_PROXY)

    def snapshot(self) -> "TraitBelief":
        return TraitBelief(alpha=dict(self.alpha), beta=dict(self.beta))

    def to_dict(self) -> Dict:
        return {
            "alpha":            dict(self.alpha),
            "beta":             dict(self.beta),
            "means":            {t: round(self.mean(t), 4)        for t in PHENOMENA},
            "uncertainties":    {t: round(self.uncertainty(t), 4) for t in PHENOMENA},
            "total_uncertainty": round(self.total_uncertainty(), 4),
        }


# =========================================================
# Data Loading
# =========================================================
def load_data_from_jsonl(
    jsonl_path: str = SNIPPET_JSONL,
    min_snippets: int = MIN_SNIPPETS_PER_UID,
) -> Dict[str, Any]:
    uid_data: Dict[str, Dict] = defaultdict(lambda: {
        "gt_traits": set(), "turns": [], "n_snippets": 0})
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            obj  = json.loads(line)
            uid  = str(obj.get("uid", ""))
            if not uid: continue
            for tid in obj.get("traits", []):
                if tid in PHENOMENA: uid_data[uid]["gt_traits"].add(tid)
            turn_id = int(obj.get("turn_id", 0))
            doc_q   = str(obj.get("doctor_curr", "")).strip()
            pat_a   = str(obj.get("patient_curr", "")).strip()
            if doc_q or pat_a:
                uid_data[uid]["turns"].append((turn_id, doc_q, pat_a))
            uid_data[uid]["n_snippets"] += 1

    result: Dict[str, Dict] = {}
    skipped = 0
    for uid, data in uid_data.items():
        n = data["n_snippets"]
        if n < min_snippets: skipped += 1; continue
        turns_sorted = sorted(data["turns"], key=lambda x: x[0])
        full_text = " ".join(
            f"DOC: {doc} PAT: {pat}"
            for _, doc, pat in turns_sorted if doc or pat)
        base, _ = parse_uid(uid)
        result[uid] = {
            "gt_traits":    sorted(data["gt_traits"]),
            "full_text":    full_text,
            "n_snippets":   n,
            "base_patient": base,
        }
    bases = {v["base_patient"] for v in result.values()}
    print(f"[load_data] uids:{len(uid_data)} | "
          f"skipped:{skipped} | retained:{len(result)} | "
          f"base_patients:{len(bases)}")
    return result


# =========================================================
# SnippetBank
# =========================================================
@dataclass
class Snippet:
    uid: str; scene_id: str; turn_id: int
    doctor_curr: str; patient_curr: str
    traits: List[str]; trait_probs: Dict[str, float]
    embedding: Optional[np.ndarray]; meta: Dict[str, Any]


class SnippetBank:
    def __init__(self, jsonl_path: str):
        self.jsonl_path = jsonl_path
        self.snippets:   List[Snippet]        = []
        self.uid_map:    Dict[str, List[int]] = defaultdict(list)

    def load(self) -> None:
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                obj = json.loads(line)
                uid = str(obj.get("uid", ""))
                if not uid: continue
                base, scene_id_str = parse_uid(uid)
                sn = Snippet(
                    uid=uid, scene_id=scene_id_str or "0",
                    turn_id=int(obj.get("turn_id", 0)),
                    doctor_curr=str(obj.get("doctor_curr", "")),
                    patient_curr=str(obj.get("patient_curr", "")),
                    traits=[str(t) for t in obj.get("traits", [])
                            if str(t) in PHENOMENA],
                    trait_probs={str(k): float(v)
                                 for k, v in obj.get("trait_probs", {}).items()},
                    embedding=None, meta={})
                idx = len(self.snippets)
                self.snippets.append(sn)
                self.uid_map[uid].append(idx)
                self.uid_map[base].append(idx)
        print(f"[SnippetBank] uids={len(self.uid_map)}, "
              f"snippets={len(self.snippets)}")

    def get_by_uid(self, uid: str) -> List[Snippet]:
        return [self.snippets[i] for i in self.uid_map.get(uid, [])]

    def estimate_theta_base(self, uid: str) -> Dict[str, float]:
        base, _ = parse_uid(uid)
        scenes  = self.get_by_uid(uid) or self.get_by_uid(base)
        if not scenes:
            return {tid: 0.15 for tid in PHENOMENA}
        counts: Dict[str, int] = defaultdict(int)
        for sn in scenes:
            for t in sn.traits: counts[t] += 1
        n = len(scenes)
        return {tid: min(0.9, max(0.05, counts[tid] / n))
                for tid in PHENOMENA}


class EmbeddingIndex:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model      = SentenceTransformer(model_name)
        self.embeddings: Optional[np.ndarray] = None
        self.snippets:   List[Snippet]         = []

    def build(self, bank: SnippetBank) -> None:
        self.snippets = bank.snippets
        texts = [f"{sn.doctor_curr} {sn.patient_curr}" for sn in self.snippets]
        print(f"[EmbeddingIndex] Encoding {len(texts)} snippets...")
        from tqdm import tqdm
        batch_size = 64
        all_embs = []
        for i in tqdm(range(0, len(texts), batch_size),
                      total=math.ceil(len(texts)/batch_size)):
            batch = texts[i:i+batch_size]
            embs  = self.model.encode(batch, convert_to_numpy=True,
                                      show_progress_bar=False)
            all_embs.append(embs)
        self.embeddings = np.vstack(all_embs)
        print(f"[EmbeddingIndex] Shape: {self.embeddings.shape}")

    def encode(self, text: str) -> np.ndarray:
        return self.model.encode([text], convert_to_numpy=True)[0]

    def query(self, strategy: str, emitted_traits: List[str],
              doctor_question: str, uid: str,
              top_k: int = 5, rng: Optional[random.Random] = None
              ) -> Optional[Snippet]:
        if self.embeddings is None: return None
        rng = rng or random.Random()
        base, _ = parse_uid(uid)
        candidate_idx = [
            i for i, sn in enumerate(self.snippets)
            if parse_uid(sn.uid)[0] != base]
        if not candidate_idx: return None
        trait_names = [TRAIT_ID_MAP.get(t, t) for t in emitted_traits]
        composite = (f"[Strategy: {strategy}] "
                     f"[Traits: {', '.join(trait_names) or 'general'}] "
                     f"{doctor_question}")
        q_emb = self.model.encode([composite], convert_to_numpy=True)[0]
        cand_embs = self.embeddings[candidate_idx]
        normed = cand_embs / (
            np.linalg.norm(cand_embs, axis=1, keepdims=True) + EPS)
        q_norm = q_emb / (np.linalg.norm(q_emb) + EPS)
        scores = normed @ q_norm
        top_idx = np.argsort(-scores)[:top_k]
        chosen = rng.choice(top_idx.tolist())
        return self.snippets[candidate_idx[chosen]]


class QuestionTraitAffinityEstimator:
    def __init__(self, emb_index: EmbeddingIndex):
        self.emb_index = emb_index
        self._centroids: Dict[str, np.ndarray] = {}
        self._build_centroids()

    def _build_centroids(self) -> None:
        buckets: Dict[str, List[np.ndarray]] = defaultdict(list)
        for i, sn in enumerate(self.emb_index.snippets):
            for tid in sn.traits:
                if self.emb_index.embeddings is not None:
                    buckets[tid].append(self.emb_index.embeddings[i])
        for tid, vecs in buckets.items():
            self._centroids[tid] = np.mean(vecs, axis=0)
        print("[AffinityEstimator] Centroids built.")

    def compute(self, q_emb: np.ndarray,
                use_patient_centroid: bool = False) -> Dict[str, float]:
        affinities: Dict[str, float] = {}
        q_norm = q_emb / (np.linalg.norm(q_emb) + EPS)
        for tid in PHENOMENA:
            if tid in self._centroids:
                c = self._centroids[tid]
                c_norm = c / (np.linalg.norm(c) + EPS)
                affinities[tid] = float(q_norm @ c_norm)
            else:
                affinities[tid] = 0.0
        return affinities


# =========================================================
# PatientLatentState + PhenotypeEmitter + HybridRealizer
# =========================================================
@dataclass
class PatientLatentState:
    theta:      Dict[str, float] = field(default_factory=dict)
    theta_base: Dict[str, float] = field(default_factory=dict)
    turn:       int = 0

    def __post_init__(self):
        for tid in PHENOMENA:
            if tid not in self.theta:      self.theta[tid]      = 0.15
            if tid not in self.theta_base: self.theta_base[tid] = 0.15


class PhenotypeEmitter:
    def __init__(self, uplift_logit_delta: Dict[str, Dict[str, float]],
                 max_emit_per_turn: int = 2):
        self.uplift_logit_delta = uplift_logit_delta
        self.max_emit_per_turn  = max_emit_per_turn

    def sample(self, z: PatientLatentState,
               strategy: str,
               q_affinity: Dict[str, float],
               rng: random.Random) -> Tuple[List[str], Dict[str, float]]:
        emitted, probs = [], {}
        candidates = sorted(PHENOMENA, key=lambda t: -q_affinity.get(t, 0))
        for tid in candidates:
            if len(emitted) >= self.max_emit_per_turn: break
            theta = z.theta.get(tid, 0.15)
            delta = self.uplift_logit_delta.get(strategy, {}).get(tid, 0.0)
            aff   = q_affinity.get(tid, 0.0)
            p     = sigmoid(logit(theta) + delta + 0.5 * aff)
            probs[tid] = round(p, 4)
            if rng.random() < p:
                emitted.append(tid)
        return emitted, probs

    def update_theta(self, z: PatientLatentState,
                     response: str,
                     affinity_estimator: QuestionTraitAffinityEstimator,
                     embedding_index:    EmbeddingIndex) -> PatientLatentState:
        r_emb      = embedding_index.encode(response)
        r_affinity = affinity_estimator.compute(r_emb)
        new_theta  = {}
        for tid in PHENOMENA:
            old = z.theta.get(tid, 0.15)
            aff = r_affinity.get(tid, 0.0)
            new_theta[tid] = min(0.9, max(0.05, old + 0.05 * (aff - 0.5)))
        return PatientLatentState(
            theta=new_theta,
            theta_base=dict(z.theta_base),
            turn=z.turn + 1)


class HybridRealizer:
    SYS = ("You are simulating a patient in an ADOS-2 Module 4 autism "
           "language assessment. Reply naturally as the patient would. "
           'Return JSON: {"patient_reply": "<reply>"}')

    def __init__(self, model: str = "gpt-4.1-nano"):
        self.llm = ChatOpenAI(model=model, temperature=0.9)

    def realize(self, doctor_question: str,
                history: List[Tuple[str, str]],
                anchor:  Optional[Snippet],
                emitted_traits: List[str]) -> Tuple[str, Dict]:
        hist_text  = "\n".join(
            f"Doctor: {_clean_text(q, 300)}\nPatient: {_clean_text(a, 300)}"
            for q, a in history[-3:])
        anchor_doc = _clean_text(anchor.doctor_curr  if anchor else "", 400)
        anchor_pat = _clean_text(anchor.patient_curr if anchor else "", 400)
        trait_names = [TRAIT_ID_MAP.get(t, t) for t in emitted_traits]

        user_prompt = (
            f"Conversation so far:\n{hist_text}\n\n"
            f"Examiner QUESTION:\n{_clean_text(doctor_question, 400)}\n\n")
        if anchor_pat:
            user_prompt += (f"ANCHOR_DOCTOR:\n{anchor_doc}\n\n"
                            f"ANCHOR_PATIENT (exhibit these language features: "
                            f"{', '.join(trait_names) or 'natural speech'}):\n"
                            f"{anchor_pat}\n\n")
        user_prompt += ("Reply as the patient. "
                        'Return JSON: {"patient_reply": "<reply>"}')
        used_fallback = False
        try:
            obj   = llm_json_call(self.llm, self.SYS, user_prompt)
            reply = obj.get("patient_reply", "").strip()
            used_fallback = obj.get("_fallback", False)
        except RuntimeError:
            reply = anchor_pat or "I am not sure how to answer that."
            used_fallback = True
        return reply or anchor_pat or "I am not sure.", {
            "used_anchor":   bool(anchor_pat),
            "used_fallback": used_fallback,
        }


# =========================================================
# TraitDetector
# =========================================================
class TraitDetector:
    TRAIT_DEFINITIONS: Dict[str, str] = {
        "1":  "F1 Echoic repetition: mimics verbatim what has been said.",
        "2":  "F2 Unconventional content: peculiarly chosen or odd phrasing.",
        "3":  "F3 Pronoun displacement: substitutes 'you' for 'I' or uses 3rd-person self.",
        "4":  "F4 Incongruous humor: inserts humour during serious discussions.",
        "5":  "F5 Formalistic language: overly formal or archaic style.",
        "6":  "F6 Superfluous phrases: attaches redundant filler expressions.",
        "7":  "F7 Excessive social phrasing: uses social expressions excessively.",
        "8":  "F8 Monotone social expression: reiterates flat social phrases.",
        "9":  "F9 Stereotyped media quoting: quotes media stereotypically.",
        "10": "F10 Clicheed verbal substitutions: uses cliches instead of direct responses.",
    }

    def __init__(self, model: str = "gpt-4.1-nano"):
        self.llm = ChatOpenAI(model=model, temperature=0.0)

    def detect(self, question: str, response: str) -> List[str]:
        knowledge = "\n".join(
            f"  {v}" for v in self.TRAIT_DEFINITIONS.values())
        prompt = (
            "You are an expert clinical linguist specialising in SLDs.\n\n"
            f"Examiner: {_clean_text(question, 400)}\n"
            f"Patient: {_clean_text(response, 400)}\n\n"
            "KNOWLEDGE -- 10 SLD categories:\n"
            f"{knowledge}\n\n"
            "Analyse PATIENT words only. "
            "Return feature numbers present, comma-separated (e.g. '2, 3, 6'). "
            "Return '0' if none. No other text.")
        try:
            raw     = self.llm.invoke([HumanMessage(content=prompt)])
            raw_txt = raw.content if hasattr(raw, "content") else str(raw)
            return _extract_trait_ids(raw_txt)
        except: return []


# =========================================================
# DoctorAgent
# =========================================================
class DoctorAgent(AssistantAgent):
    def __init__(self, name: str, system_message: str, llm_config: dict):
        super().__init__(name=name, system_message=system_message,
                         llm_config=llm_config)
        self._llm = ChatOpenAI(
            model="gpt-4.1-nano",
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.7)

    def generate_question(self, strategy: str, topic: str,
                          recent_history: str) -> str:
        """Standard / Proactive question generation (no Thought)."""
        guidance = STRATEGY_GUIDANCE.get(strategy, "Ask a clear clinical question.")
        prompt = (
            "You are conducting an autism language assessment (ADOS-2 Module 4).\n\n"
            f"Recent conversation:\n{_clean_text(recent_history, 800)}\n\n"
            f"Topic: {_clean_text(topic, 100)}\n"
            f"Strategy: {strategy}\n"
            f"Guidance: {guidance}\n\n"
            "Generate exactly ONE question. Concise and patient-friendly. "
            "Do not mention strategy names or diagnostic terms.")
        try:
            result = self._llm.invoke([HumanMessage(content=prompt)])
            q = result.content if hasattr(result, "content") else str(result)
            return (q.strip() if q and q.strip()
                    else f"Can you tell me more about {topic}?")
        except Exception:
            return f"Can you tell me more about {topic}?"


# =========================================================
# PatientAgent
# =========================================================
class PatientAgent(AssistantAgent):
    def __init__(self, uid: str, bank: SnippetBank,
                 embedding_index: EmbeddingIndex,
                 affinity_estimator: QuestionTraitAffinityEstimator,
                 uplift_logit_delta: Dict[str, Dict[str, float]],
                 llm_model: str = "gpt-4.1-nano",
                 seed: int = 123,
                 log_dir: str = LOG_DIR,
                 llm_config: Optional[dict] = None):
        self.uid                = uid
        self.bank               = bank
        self.embedding_index    = embedding_index
        self.affinity_estimator = affinity_estimator
        self._seed              = seed
        self.rng                = random.Random(seed)

        theta_base = {
            tid: min(0.55, max(0.10, float(v)))
            for tid, v in bank.estimate_theta_base(uid).items()}
        self._init_theta = theta_base

        concentration = 4.5
        self.belief = TraitBelief(
            alpha={tid: max(BELIEF_ALPHA0, theta_base[tid] * concentration)
                   for tid in PHENOMENA},
            beta ={tid: max(BELIEF_BETA0,
                            (1 - theta_base[tid]) * concentration)
                   for tid in PHENOMENA})
        self._init_belief = self.belief.snapshot()
        self.z = PatientLatentState(
            theta=dict(theta_base), theta_base=dict(theta_base))
        self.emitter  = PhenotypeEmitter(
            uplift_logit_delta=uplift_logit_delta, max_emit_per_turn=2)
        self.realizer = HybridRealizer(model=llm_model)

        self.run_dir = os.path.join(
            log_dir, "patient_turns", f"uid={uid}",
            datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(self.run_dir, exist_ok=True)

        super().__init__(
            name=f"Patient_{uid}",
            system_message="Patient simulator.",
            llm_config=llm_config or {"config_list": [{
                "model": llm_model,
                "api_key": os.getenv("OPENAI_API_KEY")}]})

    def reset(self) -> None:
        self.rng    = random.Random(self._seed)
        self.belief = self._init_belief.snapshot()
        self.z      = PatientLatentState(
            theta=dict(self._init_theta), theta_base=dict(self._init_theta))

    def respond(self, dialog_history: List[Tuple[str, str]],
                question: str, strategy: str) -> str:
        self.z.theta = self.belief.all_means()
        q_emb      = self.embedding_index.encode(question)
        q_affinity = self.affinity_estimator.compute(q_emb)
        emitted, probs = self.emitter.sample(
            self.z, strategy, q_affinity, rng=self.rng)
        anchor = self.embedding_index.query(
            strategy=strategy, emitted_traits=emitted,
            doctor_question=question, uid=self.uid,
            top_k=5, rng=self.rng)
        reply, dbg = self.realizer.realize(
            doctor_question=question, history=dialog_history,
            anchor=anchor, emitted_traits=emitted)
        z_next = self.emitter.update_theta(
            z=self.z, response=reply,
            affinity_estimator=self.affinity_estimator,
            embedding_index=self.embedding_index)
        log = {
            "uid": self.uid, "turn": self.z.turn, "strategy": strategy,
            "doctor_question": question, "patient_reply": reply,
            "emitted_traits": emitted, "emission_probs": probs,
            "belief_means":         self.belief.all_means(),
            "belief_uncertainties": self.belief.uncertainty_dict(),
        }
        with open(os.path.join(self.run_dir,
                  f"turn_{self.z.turn:03d}.json"),
                  "w", encoding="utf-8") as f:
            json.dump(log, f, ensure_ascii=False, indent=2)
        self.z = z_next
        return reply

    def generate_reply(self, messages):
        if not messages: return "(No context)"
        return self.respond([], messages[-1].get("content", ""),
                            strategy="Open-ended question")


# =========================================================
# StrategyTraitEstimator  (BED-LLM compatible cache format)
# =========================================================
class StrategyTraitEstimator:
    def __init__(self, model: str = "gpt-4.1-nano"):
        self.llm = ChatOpenAI(model=model, temperature=0.0)
        self.strategy_trait_counts: Dict[str, Dict[str, int]] = {
            s: {t: 0 for t in PHENOMENA} for s in STRATEGIES}
        self.strategy_counts: Dict[str, int] = {s: 0 for s in STRATEGIES}
        self._cache: Dict[str, str] = {}

    def _classify_strategy(self, doctor_question: str) -> str:
        if doctor_question in self._cache:
            return self._cache[doctor_question]
        sl = "\n".join(f"{i+1}. {s}" for i, s in enumerate(STRATEGIES))
        prompt = (f"Classify into exactly one strategy:\n{sl}\n\n"
                  f'Question: "{_clean_text(doctor_question, 400)}"\n\nStrategy name only.')
        try:
            raw = self.llm.invoke([HumanMessage(content=prompt)])
            raw = raw.content.strip() if hasattr(raw, "content") else ""
            matched = next(
                (s for s in STRATEGIES if s.lower() in raw.lower()),
                STRATEGIES[0])
        except:
            matched = STRATEGIES[0]
        self._cache[doctor_question] = matched
        return matched

    def fit(self, bank: SnippetBank, max_snippets: int = 2000) -> None:
        snippets = bank.snippets[:max_snippets]
        print(f"[StrategyTraitEstimator] Classifying {len(snippets)} snippets...")
        for i, sn in enumerate(snippets):
            if i % 100 == 0: print(f"  [{i}/{len(snippets)}]")
            if not sn.doctor_curr: continue
            strat = self._classify_strategy(sn.doctor_curr)
            self.strategy_counts[strat] = self.strategy_counts.get(strat, 0) + 1
            for tid in sn.traits:
                if tid in PHENOMENA:
                    self.strategy_trait_counts[strat][tid] = \
                        self.strategy_trait_counts[strat].get(tid, 0) + 1
        print("[StrategyTraitEstimator] Done.")

    def get_uplift_logit_delta(self) -> Dict[str, Dict[str, float]]:
        if hasattr(self, "_precomputed_delta"):
            return self._precomputed_delta
        total_all = sum(self.strategy_counts.values()) or 1
        count_all: Dict[str, int] = defaultdict(int)
        for s in STRATEGIES:
            for t in PHENOMENA:
                count_all[t] += self.strategy_trait_counts[s].get(t, 0)
        baseline = {t: count_all[t] / total_all for t in PHENOMENA}
        delta: Dict[str, Dict[str, float]] = {}
        for s in STRATEGIES:
            n = self.strategy_counts.get(s, 0) or 1
            delta[s] = {}
            for t in PHENOMENA:
                p_strat = self.strategy_trait_counts[s].get(t, 0) / n
                p_base  = baseline.get(t, 0.01)
                delta[s][t] = logit(max(0.01, p_strat)) - logit(max(0.01, p_base))
        return delta

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "strategy_trait_counts": self.strategy_trait_counts,
                "strategy_counts":       self.strategy_counts,
                "uplift_logit_delta":    self.get_uplift_logit_delta(),
            }, f, indent=2, ensure_ascii=False)
        print(f"[StrategyTraitEstimator] Saved -> {path}")

    def load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        if "strategy_trait_counts" in d and "strategy_counts" in d:
            self.strategy_trait_counts = {
                k: dict(v) for k, v in d["strategy_trait_counts"].items()}
            self.strategy_counts = d["strategy_counts"]
        elif "counts" in d and "totals" in d:
            self.strategy_trait_counts = {
                k: dict(v) for k, v in d["counts"].items()}
            self.strategy_counts = d["totals"]
        elif "uplift_logit_delta" in d:
            self._precomputed_delta = {
                k: dict(v) for k, v in d["uplift_logit_delta"].items()}
        else:
            raise KeyError(
                f"Unrecognised uplift cache format. Keys: {list(d.keys())}")
        print(f"[StrategyTraitEstimator] Loaded from {path}")


# =========================================================
# ProCoT Selector  -- core contribution of this file
# =========================================================
class ProCoTSelector:
    """
    ProCoT: Proactive Chain-of-Thought prompting selector.

    Implements the three prompting schemes from Deng et al. (2023):

    (1) standard    p(r | D, C)
        Ignores strategy set; doctor generates question directly.
        Selector just returns a fixed default strategy.

    (2) proactive   p(a, r | D, C, A)
        LLM selects strategy a from A given the dialogue state.
        No explicit reasoning chain -- action selection only.

    (3) procot      p(t, a, r | D, C, A)
        LLM first generates THOUGHT t (goal-state analysis),
        then selects strategy a, then generates question r.
        This is the proposed method.

    The Thought t covers (adapted from §3 and Figure 1):
      - Which SLD traits have been observed so far
      - Which traits are still uncertain / likely present but unconfirmed
      - What conversational angle would best elicit the missing traits
      - Why the selected strategy is best for that angle

    Modes:
      "procot"      -- Full ProCoT (proposed)
      "proactive"   -- Strategy selection without Thought
      "standard"    -- No strategy awareness
      "random"      -- Uniform random
      "round_robin" -- Fixed rotation
    """

    def __init__(self,
                 llm:   ChatOpenAI,
                 mode:  str = "procot",
                 seed:  int = 42):
        self.llm     = llm
        self.mode    = mode
        self._rng    = random.Random(seed)
        self._rr_idx = 0
        # Log last decision details for per-turn analysis
        self.last_thought:   str              = ""
        self.last_strategy:  str              = STRATEGIES[0]
        self.last_scores:    Dict[str, float] = {}
        self.last_score_detail: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # ProCoT: Thought → Strategy   (Eq. 3 in paper)
    # ------------------------------------------------------------------
    def _procot_select(self,
                       dialog_history: List[Tuple[str, str]],
                       known_detected: set,
                       topic:          str,
                       belief:         Optional[TraitBelief] = None) -> Tuple[str, str]:
        """
        Generate a Thought t and then select strategy a.

        Returns (thought, strategy).

        Prompt structure mirrors Figure 1(1c) / (2c) in the paper:
          - First: analyse current goal-completion status (the Thought)
          - Then:  select action from action set
        """
        recent = "\n".join(
            f"Doctor: {_clean_text(q, 250)}\nPatient: {_clean_text(a, 250)}"
            for q, a in dialog_history[-4:]) or "(start of conversation)"

        detected_desc = (
            ", ".join(f"F{t} {TRAIT_ID_MAP[t]}" for t in sorted(known_detected))
            if known_detected else "none detected yet")

        # Add belief-state info if available (uncertainty over undetected traits)
        uncertain_desc = ""
        if belief is not None:
            undetected = [t for t in PHENOMENA if t not in known_detected]
            # Sort by uncertainty descending
            undetected_sorted = sorted(
                undetected, key=lambda t: -belief.uncertainty(t))
            uncertain_desc = (
                "\nMost uncertain undetected traits: "
                + ", ".join(
                    f"F{t} {TRAIT_ID_MAP[t]} (u={belief.uncertainty(t):.2f})"
                    for t in undetected_sorted[:4]))

        prompt = (
            f"{_TASK_BACKGROUND}\n"
            f"Current topic: {_clean_text(topic, 100)}\n"
            f"SLD traits detected so far: {detected_desc}"
            f"{uncertain_desc}\n\n"
            f"Conversation history:\n{recent}\n\n"
            f"Available questioning strategies:\n{_STRATEGY_LIST}\n\n"
            "Step 1 — THOUGHT: Analyse the current assessment progress.\n"
            "  - What SLD traits have been elicited so far?\n"
            "  - Which traits are likely present but not yet confirmed?\n"
            "  - What conversational angle would best surface the missing traits?\n"
            "  - Why would a specific strategy help achieve this?\n\n"
            "Step 2 — ACTION: Based on your thought, select the single best "
            "strategy from the list above.\n\n"
            'Return JSON:\n'
            '{\n'
            '  "thought": "<your analysis>",\n'
            '  "strategy": "<exact strategy name from the list>"\n'
            '}')
        try:
            result = self.llm.invoke([HumanMessage(content=prompt)])
            raw    = result.content if hasattr(result, "content") else ""
            parsed = _try_parse_json(raw)
            if parsed:
                thought  = str(parsed.get("thought", "")).strip()
                strategy = str(parsed.get("strategy", "")).strip()
                # Match to closest known strategy
                matched = next(
                    (s for s in STRATEGIES
                     if s.lower() in strategy.lower()
                     or strategy.lower() in s.lower()),
                    None)
                if matched:
                    return thought, matched
        except Exception:
            pass
        # Fallback: uniform random
        return "No thought generated.", self._rng.choice(STRATEGIES)

    # ------------------------------------------------------------------
    # Proactive: Strategy selection only (Eq. 2 in paper)
    # ------------------------------------------------------------------
    def _proactive_select(self,
                          dialog_history: List[Tuple[str, str]],
                          known_detected: set,
                          topic:          str) -> str:
        """
        Select strategy from A without a Thought.
        Corresponds to 'Proactive Prompting' in §3.
        """
        recent = "\n".join(
            f"Doctor: {_clean_text(q, 250)}\nPatient: {_clean_text(a, 250)}"
            for q, a in dialog_history[-3:]) or "(start)"

        detected_desc = (
            ", ".join(f"F{t}" for t in sorted(known_detected))
            if known_detected else "none")

        prompt = (
            f"{_TASK_BACKGROUND}\n"
            f"Current topic: {_clean_text(topic, 100)}\n"
            f"SLD traits detected so far: {detected_desc}\n\n"
            f"Conversation history:\n{recent}\n\n"
            f"Available strategies:\n{_STRATEGY_LIST}\n\n"
            "Select the single best strategy to elicit new SLD traits. "
            'Return JSON: {"strategy": "<exact strategy name>"}')
        try:
            result = self.llm.invoke([HumanMessage(content=prompt)])
            raw    = result.content if hasattr(result, "content") else ""
            parsed = _try_parse_json(raw)
            if parsed:
                strategy = str(parsed.get("strategy", "")).strip()
                matched  = next(
                    (s for s in STRATEGIES
                     if s.lower() in strategy.lower()
                     or strategy.lower() in s.lower()),
                    None)
                if matched:
                    return matched
        except Exception:
            pass
        return self._rng.choice(STRATEGIES)

    # ------------------------------------------------------------------
    # Main interface
    # ------------------------------------------------------------------
    def select(self,
               turn:           int,
               topic:          str,
               dialog_history: List[Tuple[str, str]],
               known_detected: set,
               belief:         Optional[TraitBelief] = None
               ) -> Tuple[str, str]:
        """
        Returns (strategy, thought).
        thought is empty string for non-procot modes.
        """
        # ── Ablation baselines ──────────────────────────────────────
        if self.mode == "random":
            strategy = self._rng.choice(STRATEGIES)
            self.last_thought = ""
            self.last_strategy = strategy
            self.last_score_detail = {}
            return strategy, ""

        if self.mode == "round_robin":
            strategy = STRATEGIES[self._rr_idx % len(STRATEGIES)]
            self._rr_idx += 1
            self.last_thought = ""
            self.last_strategy = strategy
            self.last_score_detail = {}
            return strategy, ""

        if self.mode == "standard":
            # Standard prompting: no strategy selection, use default
            self.last_thought = ""
            self.last_strategy = STRATEGIES[0]
            self.last_score_detail = {}
            return STRATEGIES[0], ""

        if self.mode == "proactive":
            strategy = self._proactive_select(
                dialog_history, known_detected, topic)
            self.last_thought = ""
            self.last_strategy = strategy
            self.last_score_detail = {"selected": 1.0}
            return strategy, ""

        # ── Full ProCoT ─────────────────────────────────────────────
        thought, strategy = self._procot_select(
            dialog_history, known_detected, topic, belief)
        self.last_thought  = thought
        self.last_strategy = strategy
        self.last_score_detail = {"selected": 1.0}
        return strategy, thought


# =========================================================
# ProCoT Doctor: generates question conditioned on Thought
# =========================================================
class ProCoTDoctorAgent(DoctorAgent):
    """
    Extends DoctorAgent with a ProCoT-aware question generator.

    In ProCoT mode, the doctor question is generated conditioned
    on the Thought produced by ProCoTSelector -- this is the
    'response generation' step (r in Eq. 3) of the paper.
    """

    def generate_question_with_thought(self,
                                       strategy:       str,
                                       topic:          str,
                                       recent_history: str,
                                       thought:        str) -> str:
        """
        Generate question r conditioned on Thought t and action a.
        Corresponds to the final step of ProCoT (Eq. 3): p(r | t, a, D, C).
        """
        guidance = STRATEGY_GUIDANCE.get(strategy, "Ask a clear clinical question.")
        prompt = (
            "You are conducting an ADOS-2 Module 4 autism language assessment.\n\n"
            f"Assessment reasoning (your internal thought):\n{_clean_text(thought, 600)}\n\n"
            f"Recent conversation:\n{_clean_text(recent_history, 600)}\n\n"
            f"Topic: {_clean_text(topic, 100)}\n"
            f"Selected strategy: {strategy}\n"
            f"Strategy guidance: {guidance}\n\n"
            "Based on your reasoning above, generate exactly ONE question that:\n"
            "  1. Follows naturally from the conversation\n"
            "  2. Applies the selected strategy\n"
            "  3. Is likely to elicit the language features identified in your thought\n"
            "Concise and patient-friendly. "
            "Do not mention strategy names, diagnostic terms, or your reasoning.")
        try:
            result = self._llm.invoke([HumanMessage(content=prompt)])
            q = result.content if hasattr(result, "content") else str(result)
            return (q.strip() if q and q.strip()
                    else f"Can you tell me more about {topic}?")
        except Exception:
            return f"Can you tell me more about {topic}?"

    def generate_standard_question(self, topic: str,
                                   recent_history: str) -> str:
        """
        Standard prompting: p(r | D, C).
        No strategy, no thought -- direct question generation.
        """
        prompt = (
            "You are conducting an ADOS-2 Module 4 autism language assessment.\n\n"
            f"Recent conversation:\n{_clean_text(recent_history, 800)}\n\n"
            f"Topic: {_clean_text(topic, 100)}\n\n"
            "Generate exactly ONE clinical question to continue the assessment. "
            "Concise and patient-friendly.")
        try:
            result = self._llm.invoke([HumanMessage(content=prompt)])
            q = result.content if hasattr(result, "content") else str(result)
            return (q.strip() if q and q.strip()
                    else f"Can you tell me more about {topic}?")
        except Exception:
            return f"Can you tell me more about {topic}?"


# =========================================================
# Episode runner
# =========================================================
def run_episode(
    uid:        str,
    full_text:  str,
    gt_traits:  List[str],
    patient:    PatientAgent,
    doctor:     ProCoTDoctorAgent,
    detector:   TraitDetector,
    selector:   ProCoTSelector,
    planner_llm,
    max_turns:  int  = MAX_TURNS,
    early_stop: bool = True,
    log_dir:    str  = LOG_DIR,
) -> Dict[str, Any]:
    """
    Single episode using ProCoT prompting.
    GT NEVER visible during episode.

    Question generation per mode:
      procot      -- doctor.generate_question_with_thought(thought)
      proactive   -- doctor.generate_question(strategy)
      standard    -- doctor.generate_standard_question()
      random/rr   -- doctor.generate_question(strategy)
    """
    patient.reset()
    start_time = time.time()
    gt_set     = set(gt_traits)

    # Topic planning
    doc_questions = [_clean_text(seg.split("DOC:")[-1].strip(), 200)
                     for seg in full_text.split("PAT:") if "DOC:" in seg]
    plan_prompt = (
        "Real clinical interview questions for an autism assessment:\n"
        + " ".join(doc_questions[:20])
        + f"\n\nList exactly {max_turns} diverse language-related topics "
        "suitable for an ADOS-2 Module 4 assessment. "
        "One topic per line, no numbering.")
    try:
        plan_raw = planner_llm.invoke([HumanMessage(content=plan_prompt)])
        plan_raw = plan_raw.content if hasattr(plan_raw, "content") else str(plan_raw)
    except Exception:
        plan_raw = ""
    topic_list = [l.strip("- ").strip()
                  for l in plan_raw.strip().split("\n") if l.strip()]
    if not topic_list:
        topic_list = ["daily life", "relationships", "media",
                      "language use", "emotions"]
    while len(topic_list) < max_turns:
        topic_list += topic_list

    # Dialogue loop
    dialog_history:       List[Tuple[str, str]] = []
    detected_per_turn:    List[List[str]]        = []
    strategy_per_turn:    List[str]              = []
    thought_per_turn:     List[str]              = []
    known_detected:       set                    = set()
    actual_turns          = 0

    for turn in range(max_turns):
        # Early stopping: same criterion as BED-LLM for fair comparison
        total_unc = patient.belief.total_uncertainty()
        if early_stop and total_unc < EARLY_STOP_THRESHOLD:
            break

        topic = topic_list[turn % len(topic_list)]

        # ProCoT strategy selection (returns thought for procot mode)
        strategy, thought = selector.select(
            turn=turn, topic=topic,
            dialog_history=dialog_history,
            known_detected=known_detected,
            belief=patient.belief)
        strategy_per_turn.append(strategy)
        thought_per_turn.append(thought)

        # Question generation — mode-specific
        recent = "\n".join(
            f"Doctor: {_clean_text(q, 300)}\nPatient: {_clean_text(a, 300)}"
            for q, a in dialog_history[-2:])

        if selector.mode == "procot" and thought:
            # ProCoT: q conditioned on Thought t  (Eq. 3 final step)
            question = doctor.generate_question_with_thought(
                strategy, topic, recent, thought)
        elif selector.mode == "standard":
            # Standard: no strategy awareness  (Eq. 1)
            question = doctor.generate_standard_question(topic, recent)
        else:
            # Proactive / random / round_robin: strategy but no thought  (Eq. 2)
            question = doctor.generate_question(strategy, topic, recent)

        response = patient.respond(dialog_history, question, strategy)
        dialog_history.append((question, response))

        detected = detector.detect(question, response)
        detected_per_turn.append(detected)
        patient.belief.proxy_update(detected)
        known_detected.update(detected)
        actual_turns += 1

    # ── Post-hoc evaluation (identical across all baselines) ──────────
    n_gt    = max(1, len(gt_set))
    matched: set = set()
    fp_seen: set = set()
    coverage_by_turn:           List[float] = []
    penalized_coverage_by_turn: List[float] = []
    fp_count_by_turn:           List[int]   = []
    sr_turn_hits:   List[int] = []
    sr_hit_indices: List[int] = []
    turn_wise:      List[Dict] = []

    for turn, detected in enumerate(detected_per_turn):
        gt_hits  = [t for t in detected if t in gt_set]
        turn_hit = 1 if gt_hits else 0
        sr_turn_hits.append(turn_hit)
        if turn_hit: sr_hit_indices.append(turn + 1)
        for tid in detected:
            if tid in gt_set: matched.add(tid)
            else:             fp_seen.add(tid)
        raw_cov = len(matched) / n_gt
        fp_rate = len(fp_seen) / n_gt
        pen_cov = raw_cov * max(0.0, 1.0 - FP_PENALTY_ALPHA * fp_rate)
        coverage_by_turn.append(round(raw_cov, 4))
        penalized_coverage_by_turn.append(round(pen_cov, 4))
        fp_count_by_turn.append(len(fp_seen))
        turn_wise.append({
            "turn":            turn + 1,
            "strategy":        strategy_per_turn[turn],
            "thought":         thought_per_turn[turn],   # ProCoT-specific
            "doctor_question": dialog_history[turn][0],
            "patient_reply":   dialog_history[turn][1],
            "detected_traits": detected,
            "detected_tp":     sorted(gt_set & set(detected)),
            "detected_fp":     sorted(set(detected) - gt_set),
            "turn_hit_sr":     turn_hit,
            "belief_uncertainty_total": patient.belief.total_uncertainty(),
            "cumulative_matched":  sorted(matched),
            "cumulative_coverage": round(raw_cov, 4),
            "penalized_coverage":  round(pen_cov, 4),
            "cumulative_fp_count": len(fp_seen),
        })

    sr  = round(sum(sr_turn_hits) / max(1, len(sr_turn_hits)), 4)
    msc = (round(float(sum(sr_hit_indices) / len(sr_hit_indices)), 2)
           if sr_hit_indices else None)
    tp_w = sum(TRAIT_WEIGHTS.get(t, 1.0) for t in matched)
    fp_w = sum(TRAIT_WEIGHTS.get(t, 1.0) for t in fp_seen)
    fn_w = sum(TRAIT_WEIGHTS.get(t, 1.0) for t in (gt_set - matched))
    prec = tp_w / max(EPS, tp_w + fp_w)
    rec  = tp_w / max(EPS, tp_w + fn_w)
    f1   = 2 * prec * rec / max(EPS, prec + rec)
    aucc = (sum(coverage_by_turn) / len(coverage_by_turn)
            if coverage_by_turn else 0.0)

    turns_to_threshold: Dict[str, Optional[int]] = {}
    for thr in [0.33, 0.50, 0.67, 1.00]:
        turns_to_threshold[f"{int(thr*100)}%"] = next(
            (t + 1 for t, c in enumerate(coverage_by_turn) if c >= thr), None)

    base_patient, _ = parse_uid(uid)
    return {
        "uid":           uid,
        "base_patient":  base_patient,
        "gt_traits":     sorted(gt_traits),
        "mode":          selector.mode,
        "max_turns":     max_turns,
        "actual_turns":  actual_turns,
        "early_stopped": actual_turns < max_turns,
        "run_time_sec":  round(time.time() - start_time, 2),
        "sr":  sr, "msc": msc, "mcl": actual_turns,
        "sr_hit_turns":   sum(sr_turn_hits),
        "sr_hit_indices": sr_hit_indices,
        "turns_used":    actual_turns,
        "turns_saved":   max_turns - actual_turns,
        "coverage_by_turn":           coverage_by_turn,
        "penalized_coverage_by_turn": penalized_coverage_by_turn,
        "fp_count_by_turn":           fp_count_by_turn,
        "final_coverage": round(coverage_by_turn[-1], 4) if coverage_by_turn else 0.0,
        "final_pen_cov":  round(penalized_coverage_by_turn[-1], 4) if penalized_coverage_by_turn else 0.0,
        "final_fp_count": len(fp_seen),
        "weighted_f1":        round(f1, 4),
        "weighted_precision": round(prec, 4),
        "weighted_recall":    round(rec, 4),
        "aucc":               round(aucc, 4),
        "turns_to_threshold": turns_to_threshold,
        "matched_traits":  sorted(matched),
        "missed_traits":   sorted(gt_set - matched),
        "false_positives": sorted(fp_seen),
        "final_belief":   patient.belief.to_dict(),
        "turn_wise":      turn_wise,
        "dialog_history": [{"turn": i+1, "doctor": q, "patient": a}
                           for i, (q, a) in enumerate(dialog_history)],
        "patient_run_dir": patient.run_dir,
    }


# =========================================================
# Summary (identical across baselines)
# =========================================================
def _pad(curve: List[float], length: int) -> List[float]:
    if not curve: return [0.0] * length
    return curve + [curve[-1]] * (length - len(curve))


def compute_summary(results: List[Dict], mode: str) -> Dict[str, Any]:
    if not results: return {}
    max_t = max(len(r["coverage_by_turn"]) for r in results)

    scene_cov      = np.array([_pad(r["coverage_by_turn"], max_t)
                                for r in results])
    msc_micro_vals = [r["msc"] for r in results if r.get("msc") is not None]
    turns_micro    = [r.get("actual_turns", max_t) for r in results]

    micro = {
        "n_scenes": len(results),
        "SR":    round(float(np.mean([r["sr"]             for r in results])), 4),
        "MSC":   round(float(np.mean(msc_micro_vals)), 2) if msc_micro_vals else None,
        "MCL":   round(float(np.mean(turns_micro)), 2),
        "Cov":   round(float(np.mean([r["final_coverage"] for r in results])), 4),
        "Cov_std": round(float(np.std( [r["final_coverage"] for r in results])), 4),
        "F1":    round(float(np.mean([r["weighted_f1"]    for r in results])), 4),
        "AUCC":  round(float(np.mean([r["aucc"]           for r in results])), 4),
        "mean_coverage_by_turn": scene_cov.mean(axis=0).round(4).tolist(),
        "std_coverage_by_turn":  scene_cov.std(axis=0).round(4).tolist(),
    }

    patient_groups: Dict[str, List[Dict]] = defaultdict(list)
    for r in results:
        patient_groups[r["base_patient"]].append(r)

    patient_metrics: List[Dict] = []
    for base, scenes in patient_groups.items():
        cov_curves = np.array([_pad(s["coverage_by_turn"], max_t)
                                for s in scenes])
        msc_vals   = [s["msc"] for s in scenes if s.get("msc") is not None]
        turns_vals = [s.get("actual_turns", max_t) for s in scenes]
        patient_metrics.append({
            "base_patient":    base,
            "n_scenes":        len(scenes),
            "sr":              float(np.mean([s["sr"]             for s in scenes])),
            "msc":             float(np.mean(msc_vals)) if msc_vals else None,
            "mean_turns_used": float(np.mean(turns_vals)),
            "final_coverage":  float(np.mean([s["final_coverage"] for s in scenes])),
            "weighted_f1":     float(np.mean([s["weighted_f1"]    for s in scenes])),
            "aucc":            float(np.mean([s["aucc"]           for s in scenes])),
            "mean_coverage_by_turn": cov_curves.mean(axis=0).tolist(),
        })

    n_patients   = len(patient_metrics)
    msc_mac_vals = [p["msc"] for p in patient_metrics if p["msc"] is not None]
    pat_cov      = np.array([p["mean_coverage_by_turn"]
                              for p in patient_metrics])
    turns_mac    = [p["mean_turns_used"] for p in patient_metrics]

    macro = {
        "n_patients": n_patients,
        "SR":    round(float(np.mean([p["sr"]             for p in patient_metrics])), 4),
        "MSC":   round(float(np.mean(msc_mac_vals)), 2) if msc_mac_vals else None,
        "MCL":   round(float(np.mean(turns_mac)), 2),
        "Cov":   round(float(np.mean([p["final_coverage"] for p in patient_metrics])), 4),
        "Cov_std": round(float(np.std([p["final_coverage"] for p in patient_metrics])), 4),
        "F1":    round(float(np.mean([p["weighted_f1"]    for p in patient_metrics])), 4),
        "AUCC":  round(float(np.mean([p["aucc"]           for p in patient_metrics])), 4),
        "mean_coverage_by_turn": pat_cov.mean(axis=0).round(4).tolist(),
        "std_coverage_by_turn":  pat_cov.std(axis=0).round(4).tolist(),
    }

    return {
        "mode":        mode,
        "max_turns":   max_t,
        "turns_saved": round(max_t - float(np.mean(turns_micro)), 2),
        "n_scenes":            micro["n_scenes"],
        "n_patients":          macro["n_patients"],
        "SR":                  macro["SR"],
        "MSC":                 macro["MSC"],
        "MCL":                 macro["MCL"],
        "mean_final_coverage": macro["Cov"],
        "std_final_coverage":  macro["Cov_std"],
        "mean_weighted_f1":    macro["F1"],
        "mean_aucc":           macro["AUCC"],
        "mean_coverage_by_turn": macro["mean_coverage_by_turn"],
        "std_coverage_by_turn":  macro["std_coverage_by_turn"],
        "micro": micro,
        "macro": macro,
        "per_patient": [{
            "base_patient":    p["base_patient"],
            "n_scenes":        p["n_scenes"],
            "sr":              round(p["sr"], 4),
            "msc":             round(p["msc"], 2) if p["msc"] is not None else None,
            "mean_turns_used": round(p["mean_turns_used"], 1),
            "final_coverage":  round(p["final_coverage"], 4),
            "weighted_f1":     round(p["weighted_f1"], 4),
            "aucc":            round(p["aucc"], 4),
        } for p in sorted(patient_metrics, key=lambda x: x["base_patient"])],
    }


def save_episode_log(result: Dict, log_dir: str = LOG_DIR) -> str:
    os.makedirs(log_dir, exist_ok=True)
    ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = os.path.join(
        log_dir, f"{result['uid']}_{result['mode']}_{ts}.json")
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return fname


def save_summary(summary: Dict, label: str, log_dir: str = LOG_DIR) -> str:
    os.makedirs(log_dir, exist_ok=True)
    ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = os.path.join(log_dir, f"summary_{label}_{ts}.json")
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[Summary] Saved -> {fname}")
    return fname


def print_summary_table(summary: Dict, label: str) -> None:
    micro  = summary.get("micro", {})
    macro  = summary.get("macro", {})
    n_sc   = micro.get("n_scenes",   summary.get("n_scenes",   "?"))
    n_pat  = macro.get("n_patients", summary.get("n_patients", "?"))
    mode   = summary.get("mode", "")
    max_t  = summary.get("max_turns", "?")
    W = 100
    print(f"\n{'='*W}")
    print(f"  {label}  |  mode={mode}  "
          f"n_scenes={n_sc}  n_patients={n_pat}  max_turns={max_t}")
    print(f"\n  [MICRO — scene level, n={n_sc}]")
    print(f"  {'':22} {'SR':>7} {'MCL':>6} {'Cov':>7} {'±':>6} "
          f"{'F1':>7} {'AUCC':>7}")
    print(f"  {'-'*22} {'-'*7} {'-'*6} {'-'*7} {'-'*6} {'-'*7} {'-'*7}")
    print(f"  {'MEAN (scene)':<22} "
          f"{micro.get('SR',0):>7.1%} "
          f"{micro.get('MCL',0):>6.1f} "
          f"{micro.get('Cov',0):>7.1%} "
          f"{micro.get('Cov_std',0):>6.1%} "
          f"{micro.get('F1',0):>7.3f} "
          f"{micro.get('AUCC',0):>7.3f}")
    print(f"\n  [MACRO — patient level, n={n_pat}]")
    print(f"  {'PATIENT':<22} {'n_sc':>4} {'SR':>7} {'MCL':>6} "
          f"{'Cov':>7} {'F1':>7} {'AUCC':>7}")
    print(f"  {'-'*22} {'-'*4} {'-'*7} {'-'*6} {'-'*7} {'-'*7} {'-'*7}")
    for p in summary.get("per_patient", []):
        print(f"  {p['base_patient'][:21]:<22} {p['n_scenes']:>4} "
              f"{p['sr']:>7.1%} "
              f"{p['mean_turns_used']:>6.1f} "
              f"{p['final_coverage']:>7.1%} "
              f"{p['weighted_f1']:>7.3f} "
              f"{p['aucc']:>7.3f}")
    print(f"  {'MEAN (patient)':<22} {'':>4} "
          f"{macro.get('SR',0):>7.1%} "
          f"{macro.get('MCL',0):>6.1f} "
          f"{macro.get('Cov',0):>7.1%} "
          f"{macro.get('F1',0):>7.3f} "
          f"{macro.get('AUCC',0):>7.3f}")
    print(f"  {'STD (patient)':<22} {'':>4} {'':>7} {'':>6} "
          f"{macro.get('Cov_std',0):>7.1%} {'':>7} {'':>7}")
    saved = summary.get("turns_saved", 0)
    if saved > 0:
        print(f"  [Early stopping saved avg {saved:.1f} turns/episode]")
    print(f"{'='*W}\n")


# =========================================================
# Main Pipeline
# =========================================================
def run_pipeline(
    modes:      Optional[List[str]] = None,
    max_turns:  int  = MAX_TURNS,
    early_stop: bool = True,
    log_dir:    str  = LOG_DIR,
) -> None:
    """
    Run ProCoT pipeline with all ablation modes.

    modes: ["procot", "proactive", "standard", "random", "round_robin"]

    ProCoT ablation comparison (mirrors Table 1 in the paper):
      standard    -- baseline with no strategy awareness
      proactive   -- strategy selection without reasoning chain
      procot      -- full ProCoT: Thought → Strategy → Question  (proposed)
      random      -- uniform random strategy
      round_robin -- fixed rotation
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set.")

    if modes is None:
        modes = ["procot", "proactive", "standard", "random", "round_robin"]

    os.makedirs(log_dir, exist_ok=True)

    # Shared infrastructure
    uid_data = load_data_from_jsonl(SNIPPET_JSONL, MIN_SNIPPETS_PER_UID)
    bank     = SnippetBank(SNIPPET_JSONL)
    bank.load()
    emb_index = EmbeddingIndex("all-MiniLM-L6-v2")
    emb_index.build(bank)
    affinity_estimator = QuestionTraitAffinityEstimator(emb_index)

    # Load uplift cache -- try all known locations
    uplift_cache  = os.path.join(log_dir, "strategy_trait_uplift.json")
    bedllm_cache  = os.path.join("logs",          "strategy_trait_uplift.json")
    gdpzero_cache = os.path.join("logs_gdpzero",  "strategy_trait_uplift.json")
    estimator     = StrategyTraitEstimator("gpt-4.1-nano")
    if os.path.exists(uplift_cache):
        estimator.load(uplift_cache)
    elif os.path.exists(bedllm_cache):
        estimator.load(bedllm_cache)
        print(f"[Pipeline] Reusing BED-LLM uplift cache from {bedllm_cache}")
    elif os.path.exists(gdpzero_cache):
        estimator.load(gdpzero_cache)
        print(f"[Pipeline] Reusing GDP-Zero uplift cache from {gdpzero_cache}")
    else:
        estimator.fit(bank, max_snippets=2000)
        estimator.save(uplift_cache)
    uplift_logit_delta = estimator.get_uplift_logit_delta()

    llm_config = {"config_list": [{
        "model":   "gpt-4.1-nano",
        "api_key": os.getenv("OPENAI_API_KEY"),
    }]}
    doctor    = ProCoTDoctorAgent(
        "Doctor",
        "You conduct autism language assessments using ADOS-2 Module 4 protocols.",
        llm_config)
    detector  = TraitDetector("gpt-4.1-nano")
    planner   = ChatOpenAI(model="gpt-4.1-nano", temperature=0.5)
    cot_llm   = ChatOpenAI(model="gpt-4.1-nano", temperature=0.0,
                            api_key=os.getenv("OPENAI_API_KEY"))

    valid_uids = sorted(
        uid for uid, info in uid_data.items() if info["gt_traits"])
    n_base = len({uid_data[u]["base_patient"] for u in valid_uids})
    print(f"\n[Pipeline] {len(valid_uids)} uid-scenes | "
          f"{n_base} base patients | modes={modes}")
    print(f"[ProCoT] Prompting scheme: Thought → Strategy → Question\n")

    all_summaries: Dict[str, Dict] = {}

    for mode in modes:
        print(f"\n{'#'*65}")
        print(f"  MODE: {mode}")
        print(f"{'#'*65}\n")
        mode_log_dir = os.path.join(log_dir, mode)
        os.makedirs(mode_log_dir, exist_ok=True)
        mode_results: List[Dict] = []

        selector = ProCoTSelector(llm=cot_llm, mode=mode, seed=42)

        for uid in valid_uids:
            info      = uid_data[uid]
            gt_traits = info["gt_traits"]
            base      = info["base_patient"]
            print(f"  [{mode}] uid={uid} (patient={base}) | GT={gt_traits}")

            patient = PatientAgent(
                uid=uid, bank=bank,
                embedding_index=emb_index,
                affinity_estimator=affinity_estimator,
                uplift_logit_delta=uplift_logit_delta,
                llm_model="gpt-4.1-nano", seed=123,
                log_dir=mode_log_dir, llm_config=llm_config)

            try:
                result = run_episode(
                    uid=uid, full_text=info["full_text"],
                    gt_traits=gt_traits, patient=patient,
                    doctor=doctor, detector=detector,
                    selector=selector, planner_llm=planner,
                    max_turns=max_turns, early_stop=early_stop,
                    log_dir=mode_log_dir)

                fname = save_episode_log(result, log_dir=mode_log_dir)
                msc_s = (f"{result['msc']:.1f}"
                         if result.get("msc") is not None else "--")
                print(f"    turns={result['actual_turns']}/{max_turns} "
                      f"SR={result['sr']:.1%} MSC={msc_s} "
                      f"Cov={result['final_coverage']:.1%} "
                      f"F1={result['weighted_f1']:.3f} "
                      f"AUCC={result['aucc']:.3f}")
                mode_results.append(result)
            except Exception as e:
                print(f"    [SKIPPED] uid={uid} error={type(e).__name__}: {e}")

        summary = compute_summary(mode_results, mode)
        save_summary(summary, mode, log_dir)
        print_summary_table(summary, f"MODE: {mode}")
        all_summaries[mode] = summary

    # ── Final comparison table (mirrors Table 1 in ProCoT paper) ──────
    if len(all_summaries) > 1:
        W = 100
        print(f"\n{'='*W}")
        print("  ProCoT ABLATION COMPARISON  (MACRO / patient-level)")
        print(f"  Prompting scheme comparison: standard → proactive → procot")
        print(f"  {'MODE':<16} {'n_sc':>5} {'SR':>7} {'MCL':>6} "
              f"{'Cov':>7} {'±':>6} {'F1':>7} {'AUCC':>7}")
        print(f"  {'-'*16} {'-'*5} {'-'*7} {'-'*6} "
              f"{'-'*7} {'-'*6} {'-'*7} {'-'*7}")
        # Print in paper order: standard → proactive → procot → random → rr
        order = ["standard", "proactive", "procot", "random", "round_robin"]
        for m in order:
            if m not in all_summaries: continue
            s   = all_summaries[m]
            mac = s.get("macro", s)
            print(f"  {m:<16} {s.get('n_scenes',0):>5} "
                  f"{mac.get('SR',0):>7.1%} "
                  f"{mac.get('MCL',0):>6.1f} "
                  f"{mac.get('Cov',0):>7.1%} "
                  f"{mac.get('Cov_std',0):>6.1%} "
                  f"{mac.get('F1',0):>7.3f} "
                  f"{mac.get('AUCC',0):>7.3f}")
        print(f"{'='*W}\n")


# =========================================================
# Entry point
# =========================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="ProCoT baseline for ADOS-2 SLD trait discovery")
    parser.add_argument("--modes", nargs="+",
                        default=["procot", "proactive", "standard",
                                 "random", "round_robin"],
                        choices=["procot", "proactive", "standard",
                                 "random", "round_robin"],
                        help="Prompting modes to run")
    parser.add_argument("--max-turns",    type=int, default=MAX_TURNS)
    parser.add_argument("--no-early-stop", action="store_true")
    parser.add_argument("--log-dir",      default=LOG_DIR)
    args = parser.parse_args()

    run_pipeline(
        modes=args.modes,
        max_turns=args.max_turns,
        early_stop=not args.no_early_stop,
        log_dir=args.log_dir,
    )
