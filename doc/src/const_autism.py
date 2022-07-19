import os
import numpy as np

CLINICAL_COLUMNS = [# DIAGNOSIS RELATED
                    'diagnosis', 'asd_yn',
                    # MULLEN RELATED
                    'mullen_el','mullen_fm','mullen_rl','mullen_vr','mullen_elc_std',
                    # ADOS RELATED
                    'ados_total','ados_rrb','ados_sa',
                    # SRS RELATED
                    'srs_total_tscore','srs_social_awareness_tscore','srs_social_motivation_tscore',
                    #CBCL RELATED
                    'cbcl_scaleIV_score','cbcl_asd_score',
                    # MCHAT RELATED
                    'mchat_total','mchat_final','mchat_result']

DEMOGRAPHIC_COLUMNS = ['age', 
                       'sex',
                     'ethnicity',
                     'race',
                     'primary_education']

APP_COLUMNS = ['id', 
                'language', 
                'app_version', 
                'features_extracted', 
                'face_tracking', 
                'date', 
                'path']

CVA_COLUMNS = [# GAZE RELATED
                'BB_gaze_percent_right',
                 'BB_gaze_silhouette_score',
                 'S_gaze_percent_right',
                 'S_gaze_silhouette_score',
                 'FP_gaze_speech_correlation',
                 'FP_gaze_silhouette_score',
                 'inv_S_gaze_percent_right',
                 'mean_gaze_percent_right', #aggregated
                 'gaze_silhouette_score', #aggregated
    
                # NAME CALL RELATED
                 'proportion_of_name_call_responses',
                 'average_response_to_name_delay',
    
                # POSTURAL SWAY RELATED
                 'FB_postural_sway',
                 'FB_postural_sway_derivative',
                 'DIGC_postural_sway',
                 'DIGC_postural_sway_derivative',
                 'DIGRRL_postural_sway',
                 'DIGRRL_postural_sway_derivative',
                 'ST_postural_sway',
                 'ST_postural_sway_derivative',
                 'MP_postural_sway',
                 'MP_postural_sway_derivative',
                 'PB_postural_sway',
                 'PB_postural_sway_derivative',
                 'BB_postural_sway',
                 'BB_postural_sway_derivative',
                 'RT_postural_sway',
                 'RT_postural_sway_derivative',
                 'MML_postural_sway',
                 'MML_postural_sway_derivative',
                 'PWB_postural_sway',
                 'PWB_postural_sway_derivative',
                 'FP_postural_sway',
                 'FP_postural_sway_derivative',
                 'S_postural_sway',  #aggregated
                 'NS_postural_sway',  #aggregated
                
                # TOUCH RELATED
                 'number_of_touches',
                 'average_length',
                 'std_length',
                 'average_error',
                 'std_error',
                 'number_of_target',
                 'pop_rate',
                 'average_touch_duration',
                 'std_touch_duration',
                 'average_delay_to_pop',
                 'std_delay_to_pop',
                 'average_force_applied',
                 'std_force_applied',
                 'average_accuracy_variation',
                 'accuracy_consistency',
                 'average_touches_per_target',
                 'std_touches_per_target',
                 'average_time_spent',
                 'std_time_spent',
                 'exploratory_percentage']

DEFAULT_PREDICTORS = [# GAZE RELATED
                 'mean_gaze_percent_right', #aggregated
                 'gaze_silhouette_score', #aggregated
    
                # NAME CALL RELATED
                 'proportion_of_name_call_responses',
                 'average_response_to_name_delay',
    
                # POSTURAL SWAY RELATED
                 'S_postural_sway',  #aggregated
                 'NS_postural_sway',  #aggregated
                
                # TOUCH RELATED
                 'average_length',
                 'average_error',
                 'std_error',
                 'pop_rate',
                 'average_force_applied',
                 'std_force_applied',
                 'average_accuracy_variation',
                 'accuracy_consistency',
                 'average_touches_per_target',
                 'average_time_spent',
                 'std_time_spent',
                 'exploratory_percentage']


VALIDITY_COLUMNS = ['validity_available',
                    'completed', 
                    'StateOfTheChild', 
                    'SiblingsInTheRoom', 
                    'ShotsVaccines', 
                    'Distractions', 
                    'FamilyMemberDistract', 
                    'PetDistract', 
                    'PetNoiseDistract', 
                    'DoorbellPhoneDistract', 
                    'TVOnDistract', 
                    'OtherDistract', 
                    'SittingUp', 
                    'Hungry', 
                    'Diaper', 
                    'AppTeamComment',
                    'Comments'
]