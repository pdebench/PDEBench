# Config Documentation

This is the documentation of the config files that were used to generate the provided [pre-trained models](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-2987).
Since the default config files for all problems are already provided, this file only provides the values for the arguments that need to be changed.
N/A values mean that the default values can be used.
The complete explanation of the arguments can be found in the [README file](/README.md)

## 2D Shallow Water Equation

| Pre-trained model						| Config filename 			| ar_mode	| pushforward	| unroll_step 	| modes 	| width 	|
| :---        							| :----    					| :---		| :---			|	---:		|	---:	|	---:	| 
| 1D_diff-sorp_NA_NA_FNO.pt 			| config_diff-sorp.yaml 	| N/A 		| N/A 			| N/A			| 16		| 64		|
| 1D_diff-sorp_NA_NA_Unet-1-step.pt		| config_diff-sorp.yaml 	| False		| False			| N/A			| N/A		| N/A		|
| 1D_diff-sorp_NA_NA_Unet-AR.pt			| config_diff-sorp.yaml 	| True		| False			| N/A			| N/A		| N/A		|
| 1D_diff-sorp_NA_NA_Unet-PF-20.pt		| config_diff-sorp.yaml 	| True		| True			| 20			| N/A		| N/A		|
| 2D_diff-react_NA_NA_FNO.pt 			| config_diff-react.yaml 	| N/A 		| N/A 			| N/A			| 12		| 20		|
| 2D_diff-react_NA_NA_Unet-1-step.pt	| config_diff-react.yaml 	| False		| False			| N/A			| N/A		| N/A		|
| 2D_diff-react_NA_NA_Unet-AR.pt		| config_diff-react.yaml 	| True		| False			| N/A			| N/A		| N/A		|
| 2D_diff-react_NA_NA_Unet-PF-20.pt		| config_diff-react.yaml 	| True		| True			| 20			| N/A		| N/A		|
| 2D_rdb_NA_NA_FNO.pt					| config_rdb.yaml 			| N/A 		| N/A 			| N/A			| 12		| 20		|
| 2D_rdb_NA_NA_Unet-1-step.pt			| config_rdb.yaml 			| False		| False			| N/A			| N/A		| N/A		|
| 2D_rdb_NA_NA_Unet-AR.pt				| config_rdb.yaml 			| True		| False			| N/A			| N/A		| N/A		|
| 2D_rdb_NA_NA_Unet-PF-20.pt			| config_rdb.yaml 			| True		| True			| 20			| N/A		| N/A		|
