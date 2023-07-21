# ----- Spaun (character & instruction) presets -----
stim_presets = {}

# Standard Spaun stimulus presets
# TODO: Add in configuration options into presets as well?
stim_presets["copy_draw"] = ("A0[#1]?X", "")
stim_presets["copy_draw_mult"] = ("A0[#1#2#3]?XXX", "")
stim_presets["digit_recog"] = ("A1[#1]?XXX", "")
stim_presets["learning"] = ("A2?{X:30}", "")
stim_presets["memory_3"] = ("A3[123]?XXXX", "")
stim_presets["memory_4"] = ("A3[1234]?XXXX", "")
stim_presets["memory_7"] = ("A3[2567589]?XXXXXXXXX", "")
stim_presets["count_3"] = ("A4[5][3]?XXXXXX", "")
stim_presets["count_9"] = ("A4[0][9]?XXXXXXXXXXX", "")
stim_presets["count_3_list"] = ("A4[321][3]?XXXXXXX", "")
stim_presets["qa_kind"] = ("A5[123]K[3]?X", "")
stim_presets["qa_pos"] = ("A5[123]P[1]?X", "")
stim_presets["rvc_simple"] = ("A6[12][2][82][2][42]?XXXXX", "")
stim_presets["rvc_complex"] = ("A6[8812][12][8842][42][8862][62][8832]?XXXXX",
                               "")
stim_presets["induction_simple"] = ("A7[1][2][3][2][3][4][3][4]?X", "")
stim_presets["induction_incomplete"] = ("A7[1][2][3][2]?XX", "")
stim_presets["induction_ravens"] = ("A7[1][11][111][2][22][222][3][33]?XXXXX",
                                    "")

# Darpa adaptive motor presets
stim_presets["darpa_adapt_motor1"] = ("{A3[#4#2#7#5]?XXXX:8}", "")

# Darpa imagenet presets
stim_presets["darpa_imagenet1"] = ("{AC[#BOX_TURTLE][#BOX_TURTLE]?X" +
                                   "AC[#SEWING_MACHINE][#SEWING_MACHINE]?X" +
                                   "AC[#GUENON][#GUENON]?X" +
                                   "AC[#TIBETAN_TERRIER][#TIBETAN_TERRIER]?X" +
                                   "AC[#PERSIAN_CAT][#PERSIAN_CAT]?X:5}", "")
stim_presets["darpa_imagenet2"] = ("{AC[#BOX_TURTLE][#SEWING_MACHINE]?X" +
                                   "AC[#BOX_TURTLE][#GUENON]?X" +
                                   "AC[#BOX_TURTLE][#TIBETAN_TERRIER]?X" +
                                   "AC[#BOX_TURTLE][#PERSIAN_CAT]?X:5}", "")
stim_presets["darpa_imagenet3"] = ("{AC[#SEWING_MACHINE][#BOX_TURTLE]?X" +
                                   "AC[#SEWING_MACHINE][#GUENON]?X" +
                                   "AC[#SEWING_MACHINE][#TIBETAN_TERRIER]?X" +
                                   "AC[#SEWING_MACHINE][#PERSIAN_CAT]?X:5}",
                                   "")
stim_presets["darpa_imagenet4"] = ("{AC[#GUENON][#BOX_TURTLE]?X" +
                                   "AC[#GUENON][#SEWING_MACHINE]?X" +
                                   "AC[#GUENON][#TIBETAN_TERRIER]?X" +
                                   "AC[#GUENON][#PERSIAN_CAT]?X:5}", "")
stim_presets["darpa_imagenet5"] = ("{AC[#TIBETAN_TERRIER][#BOX_TURTLE]?X" +
                                   "AC[#TIBETAN_TERRIER][#SEWING_MACHINE]?X" +
                                   "AC[#TIBETAN_TERRIER][#GUENON]?X" +
                                   "AC[#TIBETAN_TERRIER][#PERSIAN_CAT]?X:5}",
                                   "")
stim_presets["darpa_imagenet6"] = ("{AC[#PERSIAN_CAT][#BOX_TURTLE]?X" +
                                   "AC[#PERSIAN_CAT][#SEWING_MACHINE]?X" +
                                   "AC[#PERSIAN_CAT][#GUENON]?X" +
                                   "AC[#PERSIAN_CAT][#TIBETAN_TERRIER]?X:5}",
                                   "")

# Darpa instruction following presets
stim_resp_i = "I1: VIS*ONE, DATA*POS1*NIN;I2: VIS*TWO, DATA*POS1*EIG;" + \
              "I3: VIS*THR, DATA*POS1*SEV;I4: VIS*FOR, DATA*POS1*SIX;" + \
              "I5: VIS*FIV, DATA*POS1*FIV;I6: VIS*SIX, DATA*POS1*FOR;" + \
              "I7: VIS*SEV, DATA*POS1*THR;I8: VIS*EIG, DATA*POS1*TWO;" + \
              "I9: VIS*NIN, DATA*POS1*ONE;I0: VIS*ZER, DATA*POS1*ZER"
stim_presets["darpa_instr_stim_resp_2"] = \
    ("%I1+I2%A9{?1X?2X:5}%I3+I4%A9{?4X?3X:5}", stim_resp_i)
stim_presets["darpa_instr_stim_resp_3"] = \
    ("%I1+I2+I3%A9{?1X?2X?3X:5}%I4+I5+I6%A9{?6X?5X?4X:5}", stim_resp_i)
stim_presets["darpa_instr_stim_resp_4"] = \
    ("%I1+I2+I3+I4%A9{?1X?2X?3X?4X:5}%I0+I9+I8+I7%A9{?0X?9X?8X?7X:5}",
     stim_resp_i)
stim_presets["darpa_instr_stim_resp_5"] = \
    ("%I1+I2+I3+I4+I5%A9{?1X?2X?3X?4X?5X:5}" +
     "%I0+I9+I8+I7+I6%A9{?0X?9X?8X?7X?6X:5}", stim_resp_i)
stim_presets["darpa_instr_stim_resp_6"] = \
    ("%I1+I2+I3+I4+I5+I6%A9{?1X?2X?3X?4X?5X?6X:5}" +
     "%I0+I9+I8+I7+I6+I5%A9{?0X?9X?8X?7X?6X?5X:5}", stim_resp_i)
stim_presets["darpa_instr_stim_resp_7"] = \
    ("%I1+I2+I3+I4+I5+I6+I7%A9{?1X?2X?3X?4X?5X?6X?7X:5}" +
     "%I0+I9+I8+I7+I6+I5+I4%A9{?0X?9X?8X?7X?6X?5X?4X:5}", stim_resp_i)
stim_presets["darpa_instr_stim_resp_8"] = \
    ("%I1+I2+I3+I4+I5+I6+I7+I8%A9{?1X?2X?3X?4X?5X?6X?7X?8X:5}" +
     "%I0+I9+I8+I7+I6+I5+I4+I3%A9{?0X?9X?8X?7X?6X?5X?4X?3X:5}", stim_resp_i)

stim_task_i = "I1: VIS*ONE, TASK*F;I2: VIS*TWO, TASK*C;" + \
              "I3: VIS*THR, TASK*M + DEC*REV; I4: VIS*FOR, TASK*W;" + \
              "I5: VIS*FIV, TASK*M; I6: VIS*SIX, TASK*V;" + \
              "I7: VIS*SEV, TASK*A;I8: VIS*EIG, TASK*REACT+STATE*DIRECT"
stim_presets["darpa_instr_stim_task_2"] = \
    ("%I1+I2%{M1.[1][2][3][2][3][4][3][4]?XM2.[0][3]?XXXX:5}" +
     "%I3+I4%{M3.[321]?XXXM4.[0]?X:5}", stim_task_i)
stim_presets["darpa_instr_stim_task_3"] = \
    ("%I1+I2+I3%{M1.[1][2][3][2][3][4][3][4]?XM2.[0][3]?XXXXM3.[321]?XXX:5}" +
     "%I4+I5+I6%{M4.[0]?XM5.[123]?XXXM6.[13][3][12][2][11]?X:5}", stim_task_i)
stim_presets["darpa_instr_stim_task_4"] = \
    ("%I1+I2+I3+I4%{M1.[1][2][3][2][3][4][3][4]?XM2.[0][3]?XXXX" +
     "M3.[321]?XXXM4.[9]?X:5}%I3+I4+I5+I6%{M3.[987]?XXXM4[0]?X" +
     "M5.[123]?XXXM6.[13][3][12][2][11]?X:5}", stim_task_i)
stim_presets["darpa_instr_stim_task_5"] = \
    ("%I1+I2+I3+I4+I5%{M1.[1][2][3][2][3][4][3][4]?XM2.[0][3]?XXXX" +
     "M3.[321]?XXXM4.[9]?XM5.[123]?XXX:5}%I4+I5+I6+I7+I8%{M4[0]?X" +
     "M5.[123]?XXXM6.[13][3][12][2][11]?XM7.[123]P[3]?XM8.?1X?2X:5}",
     stim_task_i)
stim_presets["darpa_instr_stim_task_6"] = \
    ("%I1+I2+I3+I4+I5+I6%{M1.[1][2][3][2][3][4][3][4]?XM2.[0][3]?XXXX" +
     "M3.[321]?XXXM4.[9]?XM5.[123]?XXXM6.[39][9][38][8][37]?X:5}" +
     "%I3+I4+I5+I6+I7+I8%{M3.[876]?XXXM4[0]?XM5.[456]?XXX" +
     "M6.[13][3][12][2][11]?XM7.[123]P[3]?XM8.?1X?2X:5}", stim_task_i)

seq_task_i = "I1: POS1, TASK*F;I2: POS2, TASK*C;" + \
             "I3: POS3, TASK*M + DEC*REV; I4: POS4, TASK*W;" + \
             "I5: POS5, TASK*M; I6: POS6, TASK*V;" + \
             "I7: POS7, TASK*A;I8: POS8, TASK*REACT+STATE*DIRECT"
stim_presets["darpa_instr_seq_task_2"] = \
    ("%I1+I2%{MP1.[1][2][3][2][3][4][3][4]?XMP2.[0][3]?XXXX:5}" +
     "%I3+I4%{MP3.[321]?XXXMP4.[0]?X:5}", seq_task_i)
stim_presets["darpa_instr_seq_task_3"] = \
    ("%I1+I2+I5%{MP1.[1][2][3][2][3][4][3][4]?XMP2.[0][3]?XXXX" +
     "MP5.[123]?XXX:5}%I3+I4+I6%{MP6.[21][1][24][4][26][6][28]?" +
     "XXMP4.[0]?XMP3.[321]?XXX:5}", seq_task_i)
stim_presets["darpa_instr_seq_task_4"] = \
    ("%I1+I2+I5+I7%{MP1.[1][2][3][2][3][4][3][4]?XMP2.[0][3]?XXXX" +
     "MP5.[123]?XXXMP7.[123]K[3]?X:5}%I3+I4+I6+I8%{MP8.?1X?2X" +
     "MP6.[21][1][24][4][26][6][28]?XXMP4.[0]?XMP3.[321]?XXX:5}", seq_task_i)
stim_presets["darpa_instr_seq_task_5"] = \
    ("%I1+I2+I5+I7+I3%{MP1.[1][2][3][2][3][4][3][4]?XMP2.[0][3]?XXXX" +
     "MP5.[123]?XXXMP7.[123]K[3]?XMP3.[456]?XXX:5}" +
     "%I3+I4+I6+I8+I2%{MP2.[0][1]?XXMP8.?1X?2X" +
     "MP6.[21][1][24][4][26][6][28]?XXMP4.[0]?XMP3.[321]?XXX:5}", seq_task_i)
stim_presets["darpa_instr_seq_task_6"] = \
    ("%I1+I2+I5+I7+I3+I4%{MP1.[1][2][3][2][3][4][3][4]?XMP2.[0][3]?XXXX" +
     "MP5.[123]?XXXMP7.[123]K[3]?XMP3.[456]?XXXMP4.[5]?X:5}" +
     "%I3+I4+I6+I8+I2+I5%{MP5.[456]?XXXMP2.[0][1]?XXMP8.?1X?2X" +
     "MP6.[21][1][24][4][26][6][28]?XXMP4.[0]?XMP3.[321]?XXX:5}", seq_task_i)

stim_presets["darpa_instr_stim_resp_demo1"] = \
    ("%I1+I2%A9?4X?9X%I1+I2+I3%A9?5XXX",
     "I1: VIS*FOR, DATA*POS1*TWO; I2: VIS*NIN, DATA*POS1*THR;" +
     "I3: VIS*FIV, DATA*(POS1*FOR + POS2*TWO + POS3*THR)")
stim_presets["darpa_instr_stim_resp_demo2"] = \
    ("%I1+I2%A9?4X?9X%I3+I4%A9?4X?9X",
     "I1: VIS*FOR, DATA*POS1*TWO; I2: VIS*NIN, DATA*POS1*THR;" +
     "I3: VIS*FOR, DATA*POS1*ONE; I4: VIS*NIN, DATA*POS1*EIG")
stim_presets["darpa_instr_stim_task_demo1"] = \
    ("%I1+I4%M1[#2]?XM2[427]?XXX",
     "I1: VIS*ONE, TASK*W; I2: VIS*TWO, TASK*R;" +
     "I3: VIS*ONE, TASK*M + DEC*FWD; I4: VIS*TWO, TASK*M + DEC*REV")
stim_presets["darpa_instr_stim_task_demo2"] = \
    ("%I1+I2%M1[<3725>]?XM2[<3725>]?X%I3+I4%M2[427]?XXX",
     "I1: VIS*ONE, TASK*W; I2: VIS*TWO, TASK*R;" +
     "I3: VIS*ONE, TASK*M + DEC*FWD; I4: VIS*TWO, TASK*M + DEC*REV")
stim_presets["darpa_instr_seq_task_demo"] = \
    ("%I1+I2+I3%MP3[<3725>]?XMP1[427]?XXXV[<3725>]?X",
     "I1: POS3, TASK*W; I2: POS2, TASK*R;I3: POS1, TASK*M + DEC*FWD")

# Darpa instruction following + imagenet + adaptive motor presets
stim_presets["darpa_combined1"] = \
    ("%I1+I2+I3%{A9?#POLICE_VAN,X?#PUCK,X?#GREY_WHALE,X:5}" +
     "%I4+I5+I6%{A9?#ORGAN,X?#GREY_WHALE,X?#HALF_TRACK,X:5}" +
     "A3[938]?XXXA3[456]?XXX",
     "I1: VIS*GREY_WHALE, DATA*POS1*NIN;" +
     "I2: VIS*POLICE_VAN, DATA*POS1*SEV;" +
     "I3: VIS*PUCK, DATA*POS1*THR;" +
     "I4: VIS*GREY_WHALE, DATA*POS1*EIG;" +
     "I5: VIS*HALF_TRACK, DATA*POS1*TWO;" +
     "I6: VIS*ORGAN, DATA*POS1*ONE")
stim_presets["darpa_combined2"] = \
    ("%I1+I2+I3%{A9?#1,X?#2,X?#3,X:5}" +
     "%I4+I5+I6%{A9?#4,X?#3,X?#5,X:5}" +
     "A3[938]?XXXA3[456]?XXX",
     "I1: VIS*ONE, DATA*POS1*NIN;" +
     "I2: VIS*TWO, DATA*POS1*SEV;" +
     "I3: VIS*THR, DATA*POS1*THR;" +
     "I4: VIS*FOR, DATA*POS1*EIG;" +
     "I5: VIS*THR, DATA*POS1*TWO;" +
     "I6: VIS*FIV, DATA*POS1*ONE")
stim_presets["darpa_combined_test"] = \
    ("%I1+I2+I3%{A9?#POLICE_VAN,X?#PUCK,X?#GREY_WHALE,X:1}",
     "I1: VIS*GREY_WHALE, DATA*POS1*NIN;" +
     "I2: VIS*POLICE_VAN, DATA*POS1*SEV;" +
     "I3: VIS*PUCK, DATA*POS1*THR;" +
     "I4: VIS*GREY_WHALE, DATA*POS1*EIG;" +
     "I5: VIS*HALF_TRACK, DATA*POS1*TWO;" +
     "I6: VIS*ORGAN, DATA*POS1*ONE")
stim_presets["darpa_combined_test2"] = \
    ("%I1+I2+I3%{A9?#POLICE_VAN,X?#PUCK,X?#GREY_WHALE,X:4}",
     "I1: VIS*GREY_WHALE, DATA*POS1*NIN;" +
     "I2: VIS*POLICE_VAN, DATA*POS1*SEV;" +
     "I3: VIS*PUCK, DATA*POS1*THR;" +
     "I4: VIS*GREY_WHALE, DATA*POS1*EIG;" +
     "I5: VIS*HALF_TRACK, DATA*POS1*TWO;" +
     "I6: VIS*ORGAN, DATA*POS1*ONE")
stim_presets["darpa_combined_test3"] = \
    ("%I1+I2+I3%{A9?#1,X?#2,X?#3,X:5}",
     "I1: VIS*ONE, DATA*POS1*NIN;" +
     "I2: VIS*TWO, DATA*POS1*SEV;" +
     "I3: VIS*THR, DATA*POS1*THR;" +
     "I4: VIS*FOR, DATA*POS1*EIG;" +
     "I5: VIS*THR, DATA*POS1*TWO;" +
     "I6: VIS*FIV, DATA*POS1*ONE")
stim_presets["CBC_combined"] = \
    ("%I1+I2+I3%{A9?#POLICE_VAN,X?#PUCK,X?#GREY_WHALE,X:1}" +
     "%I4+I5+I6%{A9?#ORGAN,X?#GREY_WHALE,X?#HALF_TRACK,X:1}",
     "I1: VIS*GREY_WHALE, DATA*POS1*NIN;" +
     "I2: VIS*POLICE_VAN, DATA*POS1*SEV;" +
     "I3: VIS*PUCK, DATA*POS1*THR;" +
     "I4: VIS*GREY_WHALE, DATA*POS1*EIG;" +
     "I5: VIS*HALF_TRACK, DATA*POS1*TWO;" +
     "I6: VIS*ORGAN, DATA*POS1*ONE")

# Thesis instruction following delayed instr stim list task (single task, changing list length)
stim_resp_i = "I1: VIS*ONE, DATA*(POS1*NIN + POS2*EIG + POS3*SEV + POS4*SIX);" + \
              "I2: VIS*TWO, DATA*(POS1*NIN + POS2*EIG + POS3*SEV + POS4*SIX + POS5*FIV);" + \
              "I3: VIS*THR, DATA*(POS1*NIN + POS2*EIG + POS3*SEV + POS4*SIX + POS5*FIV + POS6*FOR);" + \
              "I4: VIS*FOR, DATA*(POS1*NIN + POS2*EIG + POS3*SEV + POS4*SIX + POS5*FIV + POS6*FOR + POS7*THR);" + \
              "I5: VIS*FIV, DATA*(POS1*ZER + POS2*ONE + POS3*TWO + POS4*THR);" + \
              "I6: VIS*SIX, DATA*(POS1*ZER + POS2*ONE + POS3*TWO + POS4*THR + POS5*FOR);" + \
              "I7: VIS*SEV, DATA*(POS1*ZER + POS2*ONE + POS3*TWO + POS4*THR + POS5*FOR + POS6*FIV);" + \
              "I8: VIS*EIG, DATA*(POS1*ZER + POS2*ONE + POS3*TWO + POS4*THR + POS5*FOR + POS6*FIV + POS7*SIX)"
stim_presets["delay_instr_stim_resp_4"] = \
    ("%I1%{M1.?XXXXX:5}%I5%{M5.?XXXXX:5}", stim_resp_i)
stim_presets["delay_instr_stim_resp_5"] = \
    ("%I2%{M2.?XXXXXX:5}%I6%{M6.?XXXXXX:5}", stim_resp_i)
stim_presets["delay_instr_stim_resp_6"] = \
    ("%I3%{M3.?XXXXXXX:5}%I7%{M7.?XXXXXXX:5}", stim_resp_i)
stim_presets["delay_instr_stim_resp_7"] = \
    ("%I4%{M4.?XXXXXXXX:5}%I8%{M8.?XXXXXXXX:5}", stim_resp_i)

# -------------------------- Thesis Presets -----------------------------------
# Thesis instruction following custom tasks test
stim_resp_i = "I1: POS1, TASK*A;" + \
              "I2: POS2, TASK*A + STATE*QAK;" + \
              "I3: POS3, TASK*A + STATE*QAP"
stim_presets["instr_custom0"] = \
    ("%I1+I2+I3%MP1.[2764]P[3]?XV.[2]?XV.[2]?X", stim_resp_i)

stim_resp_i = "I1: POS1, TASK*V;" + \
              "I2: POS2, TASK*M;" + \
              "I3: POS3, TASK*V + STATE*TRANS1"
stim_presets["instr_custom1"] = \
    ("%I1+I2+I3%MP1.[1][3][2][4]V.[472]V.?XXX", stim_resp_i)

stim_resp_i = "I1: POS1, TASK*M;" + \
              "I2: POS2, TASK*C + STATE*CNT0;" + \
              "I3: POS3, TASK*A"
stim_presets["instr_custom2"] = \
    ("%I1+I2+I3%MP1.[326]V.[3]?XXXXXV.P[2]?X", stim_resp_i)

stim_resp_i = "I1: POS1, TASK*V;" + \
              "I2: POS2, TASK*A;" + \
              "I3: POS3, TASK*F + STATE*TRANS1;" + \
              "I4: POS4, TASK*F + STATE*TRANS2"
stim_presets["instr_custom3"] = \
    ("%I1+I2+I3+I4%MP1.[1][3][2][4]V.[472]P[2]?XXV.?XXXXV.?XX", stim_resp_i)

# Thesis demo graphs
stim_resp_i = "I1: VIS*ONE, DATA*POS1*EIG;" + \
              "I2: VIS*TWO, DATA*POS1*ONE;" + \
              "I3: VIS*ONE, DATA*POS1*TWO;" + \
              "I4: VIS*TWO, DATA*POS1*EIG"
stim_presets["instr_demo_stage2"] = \
    ("%I1+I2%A9?1X?2X%I3+I4%A9?1X?2X", stim_resp_i)
stim_presets["instr_demo_stage3"] = \
    ("%I1+I2%M1?X2?X%I3+I4%M1?X2?X", stim_resp_i)

stim_resp_i = "I1: VIS*ONE, TASK*M;" + \
              "I2: VIS*TWO, TASK*A;" + \
              "I3: VIS*ONE, TASK*C"
stim_presets["instr_demo_stage4"] = \
    ("%I1+I2%M1.[523]?XXXXM2.[679]P[2]?XX%I3%M1.[5][2]?XXXX", stim_resp_i)

stim_resp_i = "I1: POS1, TASK*C;" + \
              "I2: POS2, TASK*M"
stim_presets["instr_demo_stage5"] = \
    ("%I1+I2%MP2.[154]?XXXMP1.[6][1]?XXV.[83]?XX", stim_resp_i)


# ========================== Configuration presets ================================
cfg_presets = {}
cfg_presets["mtr_adapt_qvelff"] = ["mtr_dyn_adaptation=True",
                                   "mtr_forcefield='QVelForcefield'"]
cfg_presets["mtr_adapt_constff"] = ["mtr_dyn_adaptation=True",
                                    "mtr_forcefield='ConstForcefield'"]

cfg_presets["vis_imagenet"] = ["stim_module='imagenet'",
                               "vis_module='lif_imagenet'"]
cfg_presets["vis_imagenet_wta"] = ["stim_module='imagenet'",
                                   "vis_module='lif_imagenet_wta'"]

# Darpa adaptive motor demo configs
cfg_presets["darpa_adapt_qvelff_demo"] = \
    ["mtr_dyn_adaptation=True", "mtr_forcefield='QVelForcefield'",
     "probe_graph_config='ProbeCfgDarpaMotor'"]
cfg_presets["darpa_adapt_constff_demo"] = \
    ["mtr_dyn_adaptation=True", "mtr_forcefield='ConstForcefield'",
     "probe_graph_config='ProbeCfgDarpaMotor'"]

# Darpa imagenet demo configs
cfg_presets["darpa_vis_imagenet"] = \
    ["stim_module='imagenet'", "vis_module='lif_imagenet'",
     "probe_graph_config='ProbeCfgDarpaVisionImagenet'"]
cfg_presets["darpa_vis_imagenet_wta"] = \
    ["stim_module='imagenet'", "vis_module='lif_imagenet_wta'",
     "probe_graph_config='ProbeCfgDarpaVisionImagenet'"]

# Darpa imagenet + instruction following + adaptive motor configs
cfg_presets["darpa_combined_demo"] = \
    ["mtr_dyn_adaptation=True", "mtr_forcefield='QVelForcefield'",
     "stim_module='imagenet'", "vis_module='lif_imagenet'",
     "probe_graph_config='ProbeCfgDarpaImagenetAdaptMotor'"]
cfg_presets["darpa_combined_noadapt_demo"] = \
    ["mtr_dyn_adaptation=False", "mtr_forcefield='QVelForcefield'",
     "stim_module='imagenet'", "vis_module='lif_imagenet'",
     "probe_graph_config='ProbeCfgDarpaImagenetAdaptMotor'"]
cfg_presets["cbc_combined_noadapt_demo"] = \
    ["mtr_dyn_adaptation=False",
     "stim_module='imagenet'", "vis_module='lif_imagenet'",
     "probe_graph_config='ProbeCfgVisMtrMemSpikes'"]