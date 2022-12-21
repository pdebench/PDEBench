# Config Documentation

This is the documentation of the config files that were used to generate the provided [pre-trained models](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-2987).
Since the default config files for all problems are already provided, this file only provides the values for the arguments that need to be changed.
N/A values mean that the default values can be used.
The complete explanation of the arguments can be found in the [README file](/README.md)

| Pre-trained model								| Config filename 				| model_name| filename (data)				| ar_mode	| pushforward	| unroll_step 	| modes 	| width 	|
| :---        									| :----    						| :---		| :---							| :---		| :---			|	---:		|	---:	|	---:	| 
| 1D_diff-sorp_NA_NA_FNO.pt 					| config_diff-sorp.yaml 		| FNO		| 1D_diff-sorp_NA_NA			| N/A 		| N/A 			| N/A			| 16		| 64		|
| 1D_diff-sorp_NA_NA_Unet-1-step.pt				| config_diff-sorp.yaml 		| Unet		| 1D_diff-sorp_NA_NA			| False		| False			| N/A			| N/A		| N/A		|
| 1D_diff-sorp_NA_NA_Unet-AR.pt					| config_diff-sorp.yaml 		| Unet		| 1D_diff-sorp_NA_NA			| True		| False			| N/A			| N/A		| N/A		|
| 1D_diff-sorp_NA_NA_Unet-PF-20.pt				| config_diff-sorp.yaml 		| Unet		| 1D_diff-sorp_NA_NA			| True		| True			| 20			| N/A		| N/A		|
| 1D_diff-sorp_NA_NA_0001.h5_PINN.pt-15000.pt  	| config_pinn_diff-sorp.yaml	| PINN 		| 1D_diff-sorp_NA_NA.h5			| N/A 		| N/A 			| N/A 			| N/A 		| N/A 		|
| 1D_CFD_Shock_trans_Train_FNO.pt				| config_1DCFD.yaml				| FNO		| 1D_CFD_Shock_Eta1.e-8_Zeta1.e-8_trans_Train.hdf5		| N/A		| N/A			| N/A			| 12		| 20		|
| 1D_CFD_Shock_trans_Train_Unet.pt				| config_1DCFD.yaml				| Unet		| 1D_CFD_Shock_Eta1.e-8_Zeta1.e-8_trans_Train.hdf5		| True		| True			| 20			| N/A		| N/A		|
| ReacDiff_Nu1.0_Rho2.0_FNO.pt					| config_ReacDiff.yaml			| FNO		| ReacDiff_Nu1.0_Rho2.0.hdf5	| N/A		| N/A			| N/A			| 12		| 20		|
| ReacDiff_Nu1.0_Rho2.0_Unet.pt					| config_ReacDiff.yaml			| Unet		| ReacDiff_Nu1.0_Rho2.0.hdf5	| True		| True			| 10			| N/A		| N/A		|
| 1D_Advection_Sols_beta4.0_FNO.pt				| config_Adv.yaml				| FNO		| 1D_Advection_Sols_beta4.0.hdf5	| N/A		| N/A			| N/A			| 12		| 20		|
| 1D_Advection_Sols_beta4.0_Unet.pt				| config_Adv.yaml				| Unet		| 1D_Advection_Sols_beta4.0.hdf5	| True		| True			| 20			| N/A		| N/A		|
| 1D_Advection_Sols_beta4.0_PINN.pt-15000.pt	| config_pinn_pde1d.yaml		| PINN		| 1D_Advection_Sols_beta4.0.hdf5	| N/A		| N/A			| N/A			| N/A		| N/A		|
| 1D_Burgers_Sols_Nu1.0_FNO.pt					| config_Bgs.yaml				| FNO		| 1D_Burgers_Sols_Nu1.0.hdf5	| N/A		| N/A			| N/A			| 12		| 20		|
| 1D_Burgers_Sols_Nu1.0_Unet-PF-20.pt			| config_Bgs.yaml				| Unet		| 1D_Burgers_Sols_Nu1.0.hdf5	| True		| True			| 20			| N/A		| N/A		|
| 2D_diff-react_NA_NA_FNO.pt 					| config_diff-react.yaml 		| FNO		| 2D_diff-react_NA_NA			| N/A 		| N/A 			| N/A			| 12		| 20		|
| 2D_diff-react_NA_NA_Unet-1-step.pt			| config_diff-react.yaml 		| Unet		| 2D_diff-react_NA_NA			| False		| False			| N/A			| N/A		| N/A		|
| 2D_diff-react_NA_NA_Unet-AR.pt				| config_diff-react.yaml 		| Unet		| 2D_diff-react_NA_NA			| True		| False			| N/A			| N/A		| N/A		|
| 2D_diff-react_NA_NA_Unet-PF-20.pt				| config_diff-react.yaml 		| Unet		| 2D_diff-react_NA_NA			| True		| True			| 20			| N/A		| N/A		|
| 2D_diff-react_NA_NA_0000.h5_PINN.pt-15000.pt 	| config_pinn_diff-react.yaml 	| PINN 		| 2D_diff-react_NA_NA.h5		| N/A 		| N/A 			| N/A 			| N/A 		| N/A 		|
| 2D_rdb_NA_NA_FNO.pt							| config_rdb.yaml 				| FNO		| 2D_rdb_NA_NA					| N/A 		| N/A 			| N/A			| 12		| 20		|
| 2D_rdb_NA_NA_Unet-1-step.pt					| config_rdb.yaml 				| Unet		| 2D_rdb_NA_NA					| False		| False			| N/A			| N/A		| N/A		|
| 2D_rdb_NA_NA_Unet-AR.pt						| config_rdb.yaml 				| Unet		| 2D_rdb_NA_NA					| True		| False			| N/A			| N/A		| N/A		|
| 2D_rdb_NA_NA_Unet-PF-20.pt					| config_rdb.yaml 				| Unet		| 2D_rdb_NA_NA					| True		| True			| 20			| N/A		| N/A		|
| 2D_rdb_NA_NA_0000.h5_PINN.pt-15000.pt 		| config_pinn_swe2d.yaml 		| PINN 		| 2D_rdb_NA_NA.h5				| N/A 		| N/A 			| N/A 			| N/A 		| N/A 		|
| 2D_DarcyFlow_beta0.01_Train_FNO.pt			| config_Darcy.yaml				| FNO		| 2D_DarcyFlow_beta0.01_Train.hdf5	| N/A		| N/A			| N/A			| 12		| 20		|
| 2D_DarcyFlow_beta0.01_Train_Unet_PF_1.pt		| config_Darcy.yaml				| Unet		| 2D_DarcyFlow_beta0.01_Train.hdf5	| False		| False			| N/A			| N/A		| N/A		|
| 3D_CFD_Rand_M1.0_Eta1e-08_Zeta1e-08_periodic_Train_FNO.pt			| config_3DCFD.yaml		| FNO		| 3D_CFD_Rand_M1.0_Eta1e-08_Zeta1e-08_periodic_Train.hdf5	| N/A		| N/A			| N/A		| 12		| 20		|
| 3D_CFD_Rand_M1.0_Eta1e-08_Zeta1e-08_periodic_Train_Unet-PF-20.pt	| config_3DCFD.yaml		| Unet		| 3D_CFD_Rand_M1.0_Eta1e-08_Zeta1e-08_periodic_Train.hdf5	| True		| True			| 20		| N/A		| N/A		|