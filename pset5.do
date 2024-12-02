*******************
/* Problem Set 5 */
*******************

clear
import delimited using "train_heating_2008.txt", delimiters("    ", collapse) varnames(nonames) clear
// list in 1/10

drop v1
rename v2 fixed_cost1
rename v3 fixed_cost2
rename v4 fixed_cost3
rename v5 fixed_cost4
rename v6 fixed_cost5
rename v7 variable_cost1
rename v8 variable_cost2
rename v9 variable_cost3
rename v10 variable_cost4
rename v11 variable_cost5
rename v12 income
rename v13 choice
// describe

************
/* Part A */
************
