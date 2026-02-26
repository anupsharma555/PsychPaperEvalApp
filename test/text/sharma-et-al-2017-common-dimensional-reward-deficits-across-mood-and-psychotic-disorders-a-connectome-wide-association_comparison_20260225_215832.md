# Comparative Analysis

- runtime_seconds: 8.733
- job_status: completed
- document_id: 118
- job_id: 111

- overall_reference_points: 71
- overall_matched_points: 3
- overall_recall: 0.042
- overall_sentence_inclusion_recall: 0.042
- overall_sentence_inclusion_any_section_recall: 0.07
- overall_section_fidelity: 0.6
- overall_inclusion_precision: 0.188
- sentence_inclusion_threshold: 0.42
- match_mode: hybrid
- match_threshold: 0.42

## Diagnostic Summary
- coverage_total_gap: 55
- noisy_line_total: 0
- low_recall_sections: introduction, methods, results, discussion, conclusion
- low_sentence_inclusion_sections: introduction, methods, results, discussion, conclusion
- likely_causes:
  - Late-section extraction is weak; section boundaries likely drift after methods/results-heavy text.
  - Methods under-coverage remains high; ranking is selecting fewer protocol-detail lines than reference.
  - Current lexical matching threshold misses semantically similar lines; embedding-based matching would raise measured recall.

## INTRODUCTION
- reference_points: 12
- app_points: 2
- matched_points: 1
- recall: 0.083
- precision_proxy: 0.5
- sentence_inclusion_recall: 0.083
- sentence_inclusion_any_section_recall: 0.083
- section_fidelity: 1.0
- inclusion_precision: 0.5
- missing_top:
  - (0.138) The presence of anhedonia across mood and psychotic disorders suggests shared underlying reward system dysfunction rather than disorder-specific mechanisms alone.
  - (0.118) The National Institute of Mental Health Research Domain Criteria (RDoC) framework promotes dimensional models of psychopathology, conceptualizing anhedonia as a transdiagnostic construct linked to abnormalities in specific neural circuits.
  - (0.074) Convergent animal and human research implicates the mesolimbic reward system, particularly the ventral striatum and nucleus accumbens, in reward processing and anhedonia.
  - (0.111) Prior task-based fMRI studies show blunted ventral striatal activation during reward processing in unipolar depression, bipolar depression, schizophrenia, and psychosis-risk populations.
  - (0.143) Recent dimensional studies demonstrate that reward-related dysfunctions are observable across diagnostic boundaries rather than being confined to individual disorders.
  - (0.114) Resting-state functional connectivity studies have revealed widespread dysconnectivity in psychiatric disorders, but most prior work has used seed-based approaches that restrict analysis to predefined brain regions.
  - (0.0) Such seed-based designs inherently limit detection of broader network-level abnormalities outside selected regions.
  - (0.154) Few studies have examined reward dysfunction across multiple psychiatric diagnoses using a whole-brain, data-driven approach.

## METHODS
- reference_points: 30
- app_points: 6
- matched_points: 1
- recall: 0.033
- precision_proxy: 0.167
- sentence_inclusion_recall: 0.033
- sentence_inclusion_any_section_recall: 0.1
- section_fidelity: 0.333
- inclusion_precision: 0.167
- missing_top:
  - (0.296) The study initially assessed 244 participants; after quality assurance procedures, 225 individuals were included in final analyses.
  - (0.056) Participants were recruited into five groups: major depressive disorder (N=32), bipolar disorder (N=50), schizophrenia (N=51), psychosis risk (genetic or clinical high risk; N=39), and healthy controls (N=53).
  - (0.08) All participants underwent structured clinical diagnostic assessment, reward responsiveness assessment, and resting-state fMRI acquisition.
  - (0.071) The sample included medicated and unmedicated individuals, and medication load was quantified for secondary analyses.
  - (0.091) The Institutional Review Board approved procedures and participants provided informed consent.
  - (0.207) Reward responsivity was measured using the Behavioral Activation Scale (BAS), specifically the reward sensitivity subscale.
  - (0.214) The BAS reward sensitivity subscale has been validated as a transdiagnostic measure of reward functioning.
  - (0.138) A factor analysis was conducted within the clinical sample to confirm the reward sensitivity subfactor structure.

## RESULTS
- reference_points: 15
- app_points: 5
- matched_points: 1
- recall: 0.067
- precision_proxy: 0.2
- sentence_inclusion_recall: 0.067
- sentence_inclusion_any_section_recall: 0.067
- section_fidelity: 1.0
- inclusion_precision: 0.2
- missing_top:
  - (0.4) These regions spanned default mode network (DMN), cingulo-opercular network (CON), and mesolimbic reward circuitry.
  - (0.143) Clinical diagnosis alone did not identify reward-system hubs in MDMR analyses, emphasizing dimensional rather than categorical effects.
  - (0.16) Higher BAS reward sensitivity was associated with increased nucleus accumbens connectivity with DMN regions and decreased nucleus accumbens connectivity with cingulo-opercular regions.
  - (0.1) Within DMN clusters, higher reward sensitivity was associated with reduced DMN-DMN connectivity and increased connectivity between DMN and cingulo-opercular network.
  - (0.105) Within CON clusters, higher reward sensitivity corresponded to reduced CON-CON connectivity and increased connectivity between CON and DMN.
  - (0.316) Community detection revealed two cortical modules: default mode network module and cingulo-opercular network module.
  - (0.158) Reward deficits were associated with default mode network hyperconnectivity (t=3.75, p=2.3×10−4), decreased integration between DMN and CON (t=−5.17, p=5.3×10−7), decreased connectivity between nucleus accumbens and DMN (t=−2.45, p=1.5×10−2), and increased connectivity between nucleus accumbens and CON (t=3.35, p=9.4×10−4).
  - (0.083) Associations were present within each diagnostic category.

## DISCUSSION
- reference_points: 9
- app_points: 2
- matched_points: 0
- recall: 0.0
- precision_proxy: 0.0
- sentence_inclusion_recall: 0.0
- sentence_inclusion_any_section_recall: 0.0
- section_fidelity: 0.0
- inclusion_precision: 0.0
- missing_top:
  - (0.154) MDMR allowed unbiased exploration of the entire functional connectome without a priori seed selection.
  - (0.074) Identification of nucleus accumbens as a hub validates the data-driven method given its centrality in reward processing.
  - (0.0) Reward deficits were associated with hyperconnectivity within the dorsomedial prefrontal subsystem of the DMN.
  - (0.087) Reduced DMN-CON connectivity suggests impaired integration between internal mentation and external salience detection systems.
  - (0.0) The authors speculate that DMN hyperconnectivity may relate to rumination and difficulty transitioning from internal to external focus.
  - (0.105) Reward deficits involved reduced nucleus accumbens-DMN connectivity and increased nucleus accumbens-CON connectivity.
  - (0.0) These patterns were consistent across major depression, bipolar disorder, schizophrenia, and psychosis risk.
  - (0.067) Strengths included a large heterogeneous sample, data-driven connectome-wide methodology, dimensional framework aligned with RDoC, and replication across diagnoses.

## CONCLUSION
- reference_points: 5
- app_points: 1
- matched_points: 0
- recall: 0.0
- precision_proxy: 0.0
- sentence_inclusion_recall: 0.0
- sentence_inclusion_any_section_recall: 0.0
- section_fidelity: 0.0
- inclusion_precision: 0.0
- missing_top:
  - (0.08) Reward deficits across mood and psychotic disorders are associated with shared patterns of large-scale network dysconnectivity.
  - (0.0) Core abnormalities involve default mode network hyperconnectivity, reduced DMN-CON integration, nucleus accumbens decoupling from DMN, and increased nucleus accumbens-CON coupling.
  - (0.087) These findings support a transdiagnostic model of anhedonia grounded in corticostriatal network dysfunction.
  - (0.0) Targeting shared neural network abnormalities may inform intervention strategies for anhedonia across diagnostic categories.
  - (0.148) Longitudinal research is needed to evaluate whether modifying these network patterns promotes resilience against reward-related psychopathology.
