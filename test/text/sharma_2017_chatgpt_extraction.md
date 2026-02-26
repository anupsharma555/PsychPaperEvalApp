# INTRODUCTION

* Anhedonia, defined as diminished reward responsivity, is a central feature across multiple psychiatric disorders including major depressive disorder, bipolar disorder, and schizophrenia, and is associated with substantial psychosocial disability.
* The presence of anhedonia across mood and psychotic disorders suggests shared underlying reward system dysfunction rather than disorder-specific mechanisms alone.
* The National Institute of Mental Health Research Domain Criteria (RDoC) framework promotes dimensional models of psychopathology, conceptualizing anhedonia as a transdiagnostic construct linked to abnormalities in specific neural circuits.
* Convergent animal and human research implicates the mesolimbic reward system, particularly the ventral striatum and nucleus accumbens, in reward processing and anhedonia.
* Prior task-based fMRI studies show blunted ventral striatal activation during reward processing in unipolar depression, bipolar depression, schizophrenia, and psychosis-risk populations.
* Recent dimensional studies demonstrate that reward-related dysfunctions are observable across diagnostic boundaries rather than being confined to individual disorders.
* Resting-state functional connectivity studies have revealed widespread dysconnectivity in psychiatric disorders, but most prior work has used seed-based approaches that restrict analysis to predefined brain regions.
* Such seed-based designs inherently limit detection of broader network-level abnormalities outside selected regions.
* Few studies have examined reward dysfunction across multiple psychiatric diagnoses using a whole-brain, data-driven approach.
* The current study addresses these limitations by conducting a connectome-wide association study (CWAS) using multivariate distance-based matrix regression (MDMR) to examine the entire functional connectome in relation to dimensional reward responsiveness.
* Reward responsiveness is operationalized using the Behavioral Activation Scale (BAS) reward sensitivity subscale, consistent with RDoC dimensional principles.
* The authors hypothesized that reward deficits would be associated with common dysconnectivity patterns involving key reward regions such as the nucleus accumbens and large-scale functional networks.

# METHODS

## Participants

* The study initially assessed 244 participants; after quality assurance procedures, 225 individuals were included in final analyses.
* Participants were recruited into five groups: major depressive disorder (N=32), bipolar disorder (N=50), schizophrenia (N=51), psychosis risk (genetic or clinical high risk; N=39), and healthy controls (N=53).
* All participants underwent structured clinical diagnostic assessment, reward responsiveness assessment, and resting-state fMRI acquisition.
* The sample included medicated and unmedicated individuals, and medication load was quantified for secondary analyses.
* The Institutional Review Board approved procedures and participants provided informed consent.

## Dimensional Assessment of Reward Responsiveness

* Reward responsivity was measured using the Behavioral Activation Scale (BAS), specifically the reward sensitivity subscale.
* The BAS reward sensitivity subscale has been validated as a transdiagnostic measure of reward functioning.
* A factor analysis was conducted within the clinical sample to confirm the reward sensitivity subfactor structure.
* BAS reward sensitivity scores showed a broad distribution across diagnostic groups, supporting dimensional analysis.

## Imaging Acquisition and Preprocessing

* All imaging was performed on a Siemens 3T scanner using consistent acquisition parameters.
* Resting-state BOLD time series were corrected for image distortion and in-scanner motion, transformed into Montreal Neurological Institute (MNI) space, and downsampled to 4 mm3 resolution for computational feasibility.

## Connectome-Wide Association Study (CWAS) Using MDMR

* For each gray matter voxel, voxel-wise seed-based connectivity maps were generated using Pearson correlations between that voxel and all other gray matter voxels.
* A distance metric quantified similarity in connectivity patterns between subjects for each voxel.
* MDMR tested whether BAS reward sensitivity scores explained between-subject differences in connectivity patterns at each voxel.
* Covariates included clinical group, age, sex, and in-scanner motion.
* Statistical thresholds were controlled using cluster correction at voxel height z > 1.64 and cluster extent p < 0.01.

## Follow-Up Seed-Based Analyses

* Clusters identified via MDMR were used as seeds for traditional seed-based connectivity analyses.
* These analyses characterized the specific connections driving multivariate effects.
* The same covariates were retained in follow-up analyses.

## Network Construction and Analysis

* A graph was constructed using cortical nodes identified by MDMR.
* Subcortical regions such as nucleus accumbens were not included in cortical module detection.
* Community detection identified network modules.
* Within-network connectivity was defined as mean correlation among nodes in a module.
* Between-network connectivity was defined as mean correlation between nodes across modules.
* Associations with BAS reward sensitivity were tested using linear regression.

## Supplementary Analyses

* Within-group dimensional effects were examined.
* Analyses excluding healthy controls were performed.
* Categorical diagnostic group comparisons were conducted.
* Smoking status and composite medication load were included as additional covariates.
* Other BAS subscales and disorder-specific severity measures were evaluated for specificity.

# RESULTS

* Significant MDMR clusters were identified in left and right nucleus accumbens, temporoparietal junction (bilateral), lateral temporal cortex, insular cortex, lateral orbitofrontal cortex, and dorsomedial frontal cortex.
* These regions spanned default mode network (DMN), cingulo-opercular network (CON), and mesolimbic reward circuitry.
* Clinical diagnosis alone did not identify reward-system hubs in MDMR analyses, emphasizing dimensional rather than categorical effects.
* Higher BAS reward sensitivity was associated with increased nucleus accumbens connectivity with DMN regions and decreased nucleus accumbens connectivity with cingulo-opercular regions.
* Within DMN clusters, higher reward sensitivity was associated with reduced DMN-DMN connectivity and increased connectivity between DMN and cingulo-opercular network.
* Within CON clusters, higher reward sensitivity corresponded to reduced CON-CON connectivity and increased connectivity between CON and DMN.
* Community detection revealed two cortical modules: default mode network module and cingulo-opercular network module.
* Reward deficits were associated with default mode network hyperconnectivity (t=3.75, p=2.3×10−4), decreased integration between DMN and CON (t=−5.17, p=5.3×10−7), decreased connectivity between nucleus accumbens and DMN (t=−2.45, p=1.5×10−2), and increased connectivity between nucleus accumbens and CON (t=3.35, p=9.4×10−4).
* Associations were present within each diagnostic category.
* Effects were attenuated but directionally consistent in healthy controls.
* Excluding healthy controls strengthened associations.
* No categorical diagnostic differences in network-level summary measures were observed.
* Smoking status and medication load did not alter findings.
* Other BAS subscales showed weaker associations.
* Network effects were not related to disorder-specific illness severity measures.

# DISCUSSION

* MDMR allowed unbiased exploration of the entire functional connectome without a priori seed selection.
* Identification of nucleus accumbens as a hub validates the data-driven method given its centrality in reward processing.
* Reward deficits were associated with hyperconnectivity within the dorsomedial prefrontal subsystem of the DMN.
* Reduced DMN-CON connectivity suggests impaired integration between internal mentation and external salience detection systems.
* The authors speculate that DMN hyperconnectivity may relate to rumination and difficulty transitioning from internal to external focus.
* Reward deficits involved reduced nucleus accumbens-DMN connectivity and increased nucleus accumbens-CON connectivity.
* These patterns were consistent across major depression, bipolar disorder, schizophrenia, and psychosis risk.
* Strengths included a large heterogeneous sample, data-driven connectome-wide methodology, dimensional framework aligned with RDoC, and replication across diagnoses.
* Limitations included cross-sectional design, reduced sensitivity to localized effects with whole-brain MDMR, self-report limitations for reward sensitivity, residual medication confounding, and uncertain generalizability to addiction or ADHD.

# CONCLUSION

* Reward deficits across mood and psychotic disorders are associated with shared patterns of large-scale network dysconnectivity.
* Core abnormalities involve default mode network hyperconnectivity, reduced DMN-CON integration, nucleus accumbens decoupling from DMN, and increased nucleus accumbens-CON coupling.
* These findings support a transdiagnostic model of anhedonia grounded in corticostriatal network dysfunction.
* Targeting shared neural network abnormalities may inform intervention strategies for anhedonia across diagnostic categories.
* Longitudinal research is needed to evaluate whether modifying these network patterns promotes resilience against reward-related psychopathology.
