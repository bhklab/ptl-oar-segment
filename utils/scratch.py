# Add scratch code here
# ROIS = ["External", "GTVp", "LCTVn", "RCTVn", "Brainstem", "Esophagus",
#         "Larynx", "Cricoid_P", "OpticChiasm", "Glnd_Lacrimal_L",
#         "Glnd_Lacrimal_R", "Lens_L", "Lens_R", "Eye_L", "Eye_R",
#         "Nrv_Optic_L", "Nrv_Optic_R", "Parotid_L", "Parotid_R",
#         "SpinalCord", "Mandible_Bone", "Glnd_Submand_L",
#         "Glnd_Submand_R", "Cochlea_L", "Cochlea_R", "Lips",
#         "Spc_Retrophar_R", "Spc_Retrophar_L", "BrachialPlex_R",
#         "BrachialPlex_L", "BRAIN", "OralCavity", "Musc_Constrict_I",
#         "Musc_Constrict_S", "Musc_Constrict_M", "LEVEL_IA", "LEVEL_IB_RT",
#         "LEVEL_III_RT", "LEVEL_II_RT", "LEVEL_IV_RT", "LEVEL_VIIA_RT", "LEVEL_V_RT",
#         "LEVEL_IB_LT", "LEVEL_III_LT", "LEVEL_II_LT", "LEVEL_IV_LT", "LEVEL_VIIA_LT", "LEVEL_V_LT"]

# naming = { "Brainstem":["BRAIN_STEM"], "OpticChiasm":["CHIASM"], "Lens_L":["L_LENS", "LT_LENS"], "Lens_R": ["R_LENS", "RT_LENS"], "Eye_L": ["L_EYE", "LT_EYE"], "Eye_R": ["R_EYE", "RT_EYE"],
#     "Nrv_Optic": ["OPTICS"], "Nrv_Optic_L": ["L_OPTIC", "LT_OPTIC"], "Nrv_Optic_R": ["R_OPTIC", "RT_OPTIC"], "Parotid_L": ["LT_PAROTID", "L_PAROTID"], "Parotid_R": ["R_PAROTID", "RT_PAROTID"],
#     "Lung_L": ["L_LUNG", "LT_LUNG"], "Lung_R": ["R_LUNG", "RT_LUNG"], "Brain":["BRAIN"]}

# BRAIN_STEM.nrrd  External.nrrd  LCTVn.nrrd      LPTV56.nrrd     OPT_LPTV56.nrrd  OS_LT_PARO.nrrd    PTV70.nrrd   R_PAROTID.nrrd  SpinalCord.nrrd
# CARTILAGE.nrrd   EXT_VOLS.nrrd  L_LUNG.nrrd     LT_LUNG.nrrd    OPT_PTV70.nrrd   OS_RT_PARO.nrrd    RCTVn.nrrd   RPTV56.nrrd
# CTV70.nrrd       GTVp.nrrd      L_PAROTID.nrrd  OPT_CTV70.nrrd  OPT_RPTV56.nrrd  POST_EXT_VOL.nrrd  R_LUNG.nrrd  RT_LUNG.nrrd

# rename_ = []
# for f in folders:
#     fold = glob.glob(f+"/structures/*")
#     for c in fold:
#         if "EXTERNAL" not in c.upper():
#             oar = c.split("/")[-1]
#             if "RT_" in oar[:4]:
#                 rename_.append(f)
#                 break
#             elif "R_" in oar[:4]:
#                 rename_.append(f)
#                 break
            
# for re_ in folders:
#     fold = glob.glob(re_+"/structures/*")
#     for o in oars:
#         compare = naming[o]
#         for c in fold:
#             oar = c.split("/")[-1].partition(".")[0]
#             if oar in compare:
#                 shutil.move(c, re_+"/structures/"+o)
#                 print(oar+" moved to "+re_+"/structures/"+o)
    
#     print("Done with "+re_)
#     # break