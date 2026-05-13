# A Proactive Multi-Agent Dialogue Framework for Assessing Social Language Disorder Traits in Autism

This repository contains the datasets, prompts, and supplementary materials for the paper:  
**"A Proactive Multi-Agent Dialogue Framework for Assessing Social Language Disorder Traits in Autism"**  
by **Chuanbo Hu, Minglei Yin, Bin Liu, Wenqi Li, Lynn K. Paul, Shuo Wang, and Xin Li**.

---

## Abstract

Social Language Disorder (SLD) traits in autism spectrum disorder, characteristic linguistic behaviours including echoic repetition, pronoun displacement, and stereotyped media quoting, are largely absent from spontaneous conversation and only emerge under specific conversational conditions. In structured clinical assessments, this latency means that questioning strategy selection is a critical yet underappreciated determinant of how much diagnostic information a conversation yields. Whether large language models can be guided to proactively select questioning strategies that systematically surface these latent traits remains largely unexplored. Here we present TPA (Think, Plan, Ask), a proactive multi-agent dialogue framework applied to the language assessment component of the Autism Diagnostic Observation Schedule Module 4 (ADOS-2), in which a doctor agent explicitly reasons about which traits remain unobserved before selecting a clinically grounded strategy and generating a targeted question. A patient agent grounded in real ADOS-2 clinical data enables reproducible evaluation without real patient participation, validated across three independent experiments confirming adequate fidelity to real patient language. Evaluated on 35 patients, TPA outperforms six competitive dialogue planning baselines across all primary metrics, achieving 82.1\% SLD trait coverage. Compared with automated replay of real clinical dialogues conducted by trained clinicians, and TPA elicits 37\% more trait coverage per dialogue turn (AUCC: 0.458 VS. 0.628). These results demonstrate that proactive questioning strategy selection substantially improves the efficiency of automated SLD trait assessment, with direct implications for scalable AI-assisted clinical screening.

---

## Framework

---

## Dataset

---
## Prompt (Coarse: if existing SLD)
```text
Dialogue['text'] + "Based on the above conversation between the examiner and the patient, please categorize if any observed Social Language Disorders for the patient. Answer only 'Yes' or 'No'.“
```

---
## Prompt (Fine: which kind of SLD Features)

```text
Dialogue['text'] + "Based on the above conversation between the examiner and the patient, please categorize any observed social language disorders for the patient into the provided 10 language categories and demonstrate all instances of disorder evidence present in their dialogue.

1. Echoic Repetition: The individual mimics verbatim what has been said by others, including the examiner, or recites phrases from external sources like advertisements or movie scripts, showing a delayed echo response;
2. Unconventional Content: The speech contains peculiarly chosen content or contextually odd phrasing, such as using 'unfreshness through household' for lack of novelty, 'mideast' instead of 'midwest' for U.S. states, or describing entry into a building as 'through various apertures';
3. Pronoun Displacement: Incorrectly substitutes personal pronouns, using 'you' in place of 'I', or refers to themselves in the third person, either by pronouns like 'he/she' or by their own name;
4. Incongruous Humor Timing: Incorporates humorous or comedic expressions inappropriately during discussions meant to be serious, showing a misalignment between the content's emotional tone and the context;
5. Formalistic Language Use: Employs an overly formal or archaic language style that seems lifted from written texts, legal documents, or old literature, rather than engaging in conversational speech. Examples include elaborate ways of expressing simple ideas or feelings;
6. Superfluous Phrase Attachment: Attaches redundant phrases or filler expressions to their speech without contributing any substantive meaning or context, such as 'you know what I mean' or 'as they say,' indicating a habit rather than intentional emphasis;
7. Excessive Social Phrasing: Utilizes conventional social expressions excessively or inappropriately, responding with phrases like 'oh, thank you' in contexts where it does not fit or preempting social gestures not yet performed by the interlocutor;
8. Monotone Social Expression: Reiterates social phrases with an unchanged, monotonous intonation, indicating a lack of genuine emotional engagement or variability in social interactions;
9. Stereotyped Media Quoting: Quotes lines from commercials, movies, or TV shows in a highly stereotypical manner, employing a canned intonation that mimics the original source closely, suggesting a reliance on external media for verbal expressions;
10. Clichéd Verbal Substitutions: Resorts to well-known sayings or clichés in lieu of engaging in direct conversational responses, using phrases like 'circle of life' or 'ready to roll' as stand-ins for more personalized communication.
```
## Case Study

![Case Study Analysis: Identifying Language Deficits in an Examiner-Patient Dialogue. Phrases highlighted in indicate observed linguistic anomalies, while underscores the specific feature category of language deficits.](./images/case_study.png)
---

## Citation
If you use our data or prompts in your research, please cite our paper:
```text
@article{hu2024exploiting,
  title={Exploiting ChatGPT for Diagnosing Autism-Associated Language Disorders and Identifying Distinct Features},
  author={Hu, Chuanbo and Li, Wenqi and Ruan, Mindi and Yu, Xiangxu and Paul, Lynn K and Wang, Shuo and Li, Xin},
  journal={arXiv preprint arXiv:2405.01799},
  year={2024}
}
```
