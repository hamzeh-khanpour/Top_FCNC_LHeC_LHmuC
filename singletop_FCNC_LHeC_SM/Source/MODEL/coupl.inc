ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c      written by the UFO converter
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

C     
C     NB: VECSIZE_MEMMAX is defined in vector.inc
C     NB: vector.inc must be included before coupl.inc
C     

      DOUBLE PRECISION G, ALL_G(1)
      COMMON/STRONG/ G, ALL_G

      DOUBLE COMPLEX GAL(2)
      COMMON/WEAK/ GAL

      DOUBLE PRECISION MU_R, ALL_MU_R(1)
      COMMON/RSCALE/ MU_R, ALL_MU_R

      DOUBLE PRECISION NF
      PARAMETER(NF=0D0)
      DOUBLE PRECISION NL
      PARAMETER(NL=0D0)

      DOUBLE PRECISION MDL_MB,MDL_MC,MDL_MD,MDL_MH,MDL_MMU,MDL_MS
     $ ,MDL_MT,MDL_MTA,MDL_MU,MDL_MW,MDL_MZ,MDL_ME

      COMMON/MASSES/ MDL_MB,MDL_MC,MDL_MD,MDL_MH,MDL_MMU,MDL_MS,MDL_MT
     $ ,MDL_MTA,MDL_MU,MDL_MW,MDL_MZ,MDL_ME


      DOUBLE PRECISION MDL_WH,MDL_WT,MDL_WW,MDL_WZ

      COMMON/WIDTHS/ MDL_WH,MDL_WT,MDL_WW,MDL_WZ


      DOUBLE COMPLEX GC_1, GC_2, GC_3, GC_4, GC_52, GC_62, GC_63,
     $  GC_64, GC_68, GC_69, GC_73, GC_75, GC_76, GC_77, GC_78, GC_80,
     $  GC_83, GC_84, GC_85, GC_86, GC_87, GC_88, GC_89, GC_90, GC_91,
     $  GC_92, GC_10, GC_12, GC_14, GC_16

      COMMON/COUPLINGS/ GC_1, GC_2, GC_3, GC_4, GC_52, GC_62, GC_63,
     $  GC_64, GC_68, GC_69, GC_73, GC_75, GC_76, GC_77, GC_78, GC_80,
     $  GC_83, GC_84, GC_85, GC_86, GC_87, GC_88, GC_89, GC_90, GC_91,
     $  GC_92, GC_10, GC_12, GC_14, GC_16

