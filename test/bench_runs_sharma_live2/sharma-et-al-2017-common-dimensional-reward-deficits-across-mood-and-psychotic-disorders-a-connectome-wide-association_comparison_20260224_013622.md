# Comparative Analysis

- runtime_seconds: 0.15
- job_status: completed
- document_id: 0
- job_id: 0

- overall_reference_points: 71
- overall_matched_points: 6
- overall_recall: 0.085
- match_mode: hybrid
- match_threshold: 0.42

## Diagnostic Summary
- coverage_total_gap: 41
- noisy_line_total: 8
- low_recall_sections: introduction, methods, discussion, conclusion
- likely_causes:
  - Late-section extraction is weak; section boundaries likely drift after methods/results-heavy text.
  - Methods under-coverage remains high; ranking is selecting fewer protocol-detail lines than reference.
  - PDF artifact noise (headers/table strings/control chars) is entering ranked sentences and displacing key content.
  - Current lexical matching threshold misses semantically similar lines; embedding-based matching would raise measured recall.

## INTRODUCTION
- reference_points: 12
- app_points: 4
- matched_points: 1
- recall: 0.083
- precision_proxy: 0.25
- missing_top:
  - (0.4) Anhedonia, defined as diminished reward responsivity, is a central feature across multiple psychiatric disorders including major depressive disorder, bipolar disorder, and schizophrenia, and is associated with substantial psychosocial disability.
  - (0.235) The National Institute of Mental Health Research Domain Criteria (RDoC) framework promotes dimensional models of psychopathology, conceptualizing anhedonia as a transdiagnostic construct linked to abnormalities in specific neural circuits.
  - (0.214) Convergent animal and human research implicates the mesolimbic reward system, particularly the ventral striatum and nucleus accumbens, in reward processing and anhedonia.
  - (0.067) Prior task-based fMRI studies show blunted ventral striatal activation during reward processing in unipolar depression, bipolar depression, schizophrenia, and psychosis-risk populations.
  - (0.138) Recent dimensional studies demonstrate that reward-related dysfunctions are observable across diagnostic boundaries rather than being confined to individual disorders.
  - (0.111) Resting-state functional connectivity studies have revealed widespread dysconnectivity in psychiatric disorders, but most prior work has used seed-based approaches that restrict analysis to predefined brain regions.
  - (0.0) Such seed-based designs inherently limit detection of broader network-level abnormalities outside selected regions.
  - (0.296) Few studies have examined reward dysfunction across multiple psychiatric diagnoses using a whole-brain, data-driven approach.

## METHODS
- reference_points: 30
- app_points: 8
- matched_points: 2
- recall: 0.067
- precision_proxy: 0.25
- missing_top:
  - (0.118) All participants underwent structured clinical diagnostic assessment, reward responsiveness assessment, and resting-state fMRI acquisition.
  - (0.261) The sample included medicated and unmedicated individuals, and medication load was quantified for secondary analyses.
  - (0.032) The Institutional Review Board approved procedures and participants provided informed consent.
  - (0.291) Reward responsivity was measured using the Behavioral Activation Scale (BAS), specifically the reward sensitivity subscale.
  - (0.148) The BAS reward sensitivity subscale has been validated as a transdiagnostic measure of reward functioning.
  - (0.318) A factor analysis was conducted within the clinical sample to confirm the reward sensitivity subfactor structure.
  - (0.175) BAS reward sensitivity scores showed a broad distribution across diagnostic groups, supporting dimensional analysis.
  - (0.357) All imaging was performed on a Siemens 3T scanner using consistent acquisition parameters.

## RESULTS
- reference_points: 15
- app_points: 7
- matched_points: 3
- recall: 0.2
- precision_proxy: 0.286
- missing_top:
  - (0.222) Significant MDMR clusters were identified in left and right nucleus accumbens, temporoparietal junction (bilateral), lateral temporal cortex, insular cortex, lateral orbitofrontal cortex, and dorsomedial frontal cortex.
  - (0.4) These regions spanned default mode network (DMN), cingulo-opercular network (CON), and mesolimbic reward circuitry.
  - (0.098) Clinical diagnosis alone did not identify reward-system hubs in MDMR analyses, emphasizing dimensional rather than categorical effects.
  - (0.231) Within CON clusters, higher reward sensitivity corresponded to reduced CON-CON connectivity and increased connectivity between CON and DMN.
  - (0.333) Community detection revealed two cortical modules: default mode network module and cingulo-opercular network module.
  - (0.065) Associations were present within each diagnostic category.
  - (0.0) Effects were attenuated but directionally consistent in healthy controls.
  - (0.0) Excluding healthy controls strengthened associations.

## DISCUSSION
- reference_points: 9
- app_points: 6
- matched_points: 0
- recall: 0.0
- precision_proxy: 0.0
- missing_top:
  - (0.261) MDMR allowed unbiased exploration of the entire functional connectome without a priori seed selection.
  - (0.222) Identification of nucleus accumbens as a hub validates the data-driven method given its centrality in reward processing.
  - (0.194) Reward deficits were associated with hyperconnectivity within the dorsomedial prefrontal subsystem of the DMN.
  - (0.103) Reduced DMN-CON connectivity suggests impaired integration between internal mentation and external salience detection systems.
  - (0.0) The authors speculate that DMN hyperconnectivity may relate to rumination and difficulty transitioning from internal to external focus.
  - (0.286) Reward deficits involved reduced nucleus accumbens-DMN connectivity and increased nucleus accumbens-CON connectivity.
  - (0.027) These patterns were consistent across major depression, bipolar disorder, schizophrenia, and psychosis risk.
  - (0.074) Strengths included a large heterogeneous sample, data-driven connectome-wide methodology, dimensional framework aligned with RDoC, and replication across diagnoses.

## CONCLUSION
- reference_points: 5
- app_points: 8
- matched_points: 0
- recall: 0.0
- precision_proxy: 0.0
- missing_top:
  - (0.353) Reward deficits across mood and psychotic disorders are associated with shared patterns of large-scale network dysconnectivity.
  - (0.182) Core abnormalities involve default mode network hyperconnectivity, reduced DMN-CON integration, nucleus accumbens decoupling from DMN, and increased nucleus accumbens-CON coupling.
  - (0.182) These findings support a transdiagnostic model of anhedonia grounded in corticostriatal network dysfunction.
  - (0.32) Targeting shared neural network abnormalities may inform intervention strategies for anhedonia across diagnostic categories.
  - (0.286) Longitudinal research is needed to evaluate whether modifying these network patterns promotes resilience against reward-related psychopathology.
