# Dataset Description
## Overview
This dataset contains information about house prices in the US from 2006 to 2010, inclusive, with 1,460 entries.

## Explanation of the columns for the house price dataset:

| Column          | Explanation                                               |
|-----------------|-----------------------------------------------------------|
| MSSubClass      | Building class/type of the house (categorical).           |
| MSZoning        | General zoning classification of the land (residential, commercial, etc.). |
| LotFrontage     | Linear feet of street connected to the property.          |
| LotArea         | Total area of the lot in square feet.                      |
| Street          | Type of road access to the property (paved, gravel, etc.).|
| LotShape        | Shape of the lot (e.g., rectangular, irregular).           |
| LandContour     | Flatness of the land (level, slope).                      |
| Utilities       | Availability of utilities (electricity, water).            |
| LotConfig       | Lot configuration within the subdivision.                  |
| LandSlope       | Slope of the land (gentle, steep).                         |
| Neighborhood    | Physical location within the city/neighborhood.            |
| Condition1      | Proximity to major road or railroad.                       |
| Condition2      | Secondary condition/location (additional context).         |
| BldgType        | Type of building (detached, townhouse, etc.).              |
| HouseStyle      | Style of the house (one-story, two-story).                 |
| OverallQual     | Overall material and finish quality (rating).              |
| OverallCond     | Overall condition rating.                                   |
| YearBuilt       | Year the house was originally built.                        |
| YearRemodAdd    | Year remodeled or added.                                    |
| RoofStyle       | Type of roof.                                              |
| RoofMatl        | Roof material.                                             |
| Exterior1st     | Exterior covering on the house.                            |
| Exterior2nd     | Secondary exterior covering.                               |
| MasVnrType      | Masonry veneer type.                                       |
| MasVnrArea      | Area of masonry veneer in square feet.                     |
| ExterQual       | Exterior quality rating.                                   |
| ExterCond       | Exterior condition rating.                                 |
| Foundation      | Foundation type.                                           |
| BsmtQual        | Basement height/quality rating.                            |
| BsmtCond        | Basement condition rating.                                 |
| BsmtExposure    | Walkout or garden level basement walls.                    |
| BsmtFinType1    | Quality of basement finished area.                         |
| BsmtFinSF1      | Square feet of basement finished area type 1.              |
| BsmtFinType2    | Quality of basement finished area type 2.                  |
| BsmtFinSF2      | Square feet of basement finished area type 2.              |
| BsmtUnfSF       | Unfinished basement area in square feet.                   |
| TotalBsmtSF     | Total basement area square feet.                          |
| Heating         | Type of heating.                                          |
| HeatingQC       | Heating quality and condition.                           |
| CentralAir      | Central air conditioning (yes/no).                        |
| Electrical      | Electrical system type.                                  |
| 1stFlrSF        | First floor square feet.                                 |
| 2ndFlrSF        | Second-floor square feet.                               |
| LowQualFinSF    | Low quality finished square feet.                       |
| GrLivArea       | Above ground living area square feet.                    |
| BsmtFullBath    | Basement full bathrooms count.                           |
| BsmtHalfBath    | Basement half bathrooms count.                         |
| FullBath        | Full bathrooms above ground.                            |
| HalfBath        | Half bathrooms above ground.                            |
| BedroomAbvGr    | Number of bedrooms above ground.                        |
| KitchenAbvGr    | Number of kitchens above ground.                        |
| KitchenQual     | Kitchen quality rating.                                |
| TotRmsAbvGrd    | Total rooms above ground (excl. bathrooms).            |
| Functional      | Home functionality rating.                             |
| Fireplaces      | Number of fireplaces.                                  |
| FireplaceQu     | Fireplace quality rating.                             |
| GarageType      | Garage location/type.                                 |
| GarageYrBlt     | Year garage was built.                               |
| GarageFinish    | Interior finish of the garage.                      |
| GarageCars      | Size of garage in car capacity.                    |
| GarageArea      | Garage area in square feet.                       |
| GarageQual      | Garage quality rating.                            |
| GarageCond      | Garage condition rating.                          |
| PavedDrive     | Paved driveway status (paved, gravel).            |
| WoodDeckSF     | Wood deck area in square feet.                     |
| OpenPorchSF      | Open porch area in square feet.                    |
| EnclosedPorch   | Enclosed porch area.                              |
| 3SsnPorch      | Three-season porch area.                          |
| ScreenPorch    | Screened porch area.                              |
| PoolArea      | Pool area in square feet.                          |
| MiscVal      | Miscellaneous value (not covered by other categories).   |
| MoSold       | Month house sold.                                         |
| YrSold       | Year house sold.                                          |
| SaleType     | Type of sale.                                             |
| SaleCondition | Condition of sale.                                        |
| SalePrice    | Sale price of the property (target variable to predict). |

## Explanation for each columns with less than 50 unique values: 61 / 81
**1. Number of non-null rows of column MSSubClass: 1460 / 1460**

| MSSubClass | Count | Explanation                                               |
|------------|--------|-----------------------------------------------------------|
| 20         | 536    | 1-STORY 1946 & NEWER ALL STYLES — One-story houses built in 1946 or later, any architectural style. |
| 60         | 299    | 2-STORY 1946 & NEWER — Two-story houses built in 1946 or later. |
| 50         | 144    | 1-1/2 STORY FINISHED ALL AGES — One and a half story houses with finished living space. |
| 120        | 87     | 1-STORY PUD (Planned Unit Development) - 1946 & NEWER — One-story houses in planned unit developments built in 1946 or later. |
| 30         | 69     | 1-STORY 1945 & OLDER — One-story houses built in 1945 or earlier. |
| 160        | 63     | 2-STORY PUD - 1946 & NEWER — Two-story houses in planned unit developments built in 1946 or later. |
| 70         | 60     | 2-STORY 1945 & OLDER — Two-story houses built in 1945 or earlier. |
| 80         | 58     | SPLIT OR MULTI-LEVEL — Houses with split or multiple levels. |
| 90         | 52     | DUPLEX - ALL STYLES AND AGES — Duplex homes of any style or age. |
| 190        | 30     | 2 FAMILY CONVERSION - ALL STYLES AND AGES — Houses converted into two-family dwellings. |
| 85         | 20     | SPLIT FOYER — Split foyer houses. |
| 75         | 16     | 2-1/2 STORY ALL AGES — Two and a half story houses of any age. |
| 45         | 12     | 1-1/2 STORY - UNFINISHED ALL AGES — One and a half story houses with unfinished living space. |
| 180        | 10     | PUD - MULTI-LEVEL - INCL SPLIT LEVEL & FOYER — Planned unit developments with multi-level houses including split-level and foyer styles. |
| 40         | 4      | 1-STORY W/FINISHED ATTIC ALL AGES — One-story houses with finished attic. |


**2. Number of non-null rows of column MSZoning: 1460 / 1460**

| MSZoning  | Count | Explanation                                                       |
|-----------|--------|-------------------------------------------------------------------|
| RL        | 1151   | Residential Low Density — Single-family homes on larger lots, typically suburban or rural neighborhoods. |
| RM        | 218    | Residential Medium Density — Medium-density residential areas including townhomes or smaller lots. |
| FV        | 65     | Floating Village Residential — Higher-end residential areas often near water or scenic locations. |
| RH        | 16     | Residential High Density — High-density zones, allowing smaller lots or multifamily dwellings. |
| C (all)   | 10     | Commercial — Areas zoned for commercial use including retail, offices, and other businesses. |

**5. Street (1460 / 1460 non-null rows)**

| Value | Count | Explanation                                           |
|-------|-------|-------------------------------------------------------|
| Pave  | 1454  | Paved street access with asphalt or concrete surface.|
| Grvl  | 6     | Gravel street access, common in rural or less developed areas. |

**6. Alley (91 / 1460 non-null rows)**

| Value | Count | Explanation                           |
|-------|-------|-------------------------------------|
| Grvl  | 50    | Gravel alley access behind the house.|
| Pave  | 41    | Paved alley access behind the house.|

**7. LotShape (1460 / 1460 non-null rows)**

| Value | Count | Explanation                    |
|-------|-------|--------------------------------|
| Reg   | 925   | Regular, rectangular lot shape.|
| IR1   | 484   | Slightly irregular lot shape.  |
| IR2   | 41    | Moderately irregular lot shape.|
| IR3   | 10    | Severely irregular lot shape.  |

**8. LandContour (1460 / 1460 non-null rows)**

| Value | Count | Explanation                    |
|-------|-------|-------------------------------|
| Lvl   | 1311  | Level terrain, flat land.      |
| Bnk   | 63    | Land slopes down on one side.  |
| HLS   | 50    | Hillside terrain.              |
| Low   | 36    | Low-lying land, potentially flood-prone. |

**9. Utilities (1460 / 1460 non-null rows)**

| Value  | Count | Explanation                           |
|--------|-------|-------------------------------------|
| AllPub | 1459  | All public utilities available (electricity, gas, water, sewer). |
| NoSeWa | 1     | No public sewer or water available. |

**10. LotConfig (1460 / 1460 non-null rows)**

| Value   | Count | Explanation                                    |
|---------|-------|------------------------------------------------|
| Inside  | 1052  | Lot surrounded by other lots (interior lot).  |
| Corner  | 263   | Corner lot at street intersection.            |
| CulDSac | 94    | Lot located on a cul-de-sac (dead-end street).|
| FR2     | 47    | Frontage on two sides of the property.         |
| FR3     | 4     | Frontage on three sides of the property.       |

**11. LandSlope (1460 / 1460 non-null rows)**

| Value | Count | Explanation         |
|-------|-------|---------------------|
| Gtl   | 1382  | Gentle slope.       |
| Mod   | 65    | Moderate slope.     |
| Sev   | 13    | Severe slope.       |

**12. Neighborhood (1460 / 1460 non-null rows)**

| Value    | Count | Explanation                                      |
|----------|-------|-------------------------------------------------|
| NAmes    | 225   | North Ames neighborhood                          |
| CollgCr  | 150   | College Creek neighborhood                       |
| OldTown  | 113   | Old Town neighborhood                            |
| Edwards  | 100   | Edwards neighborhood                             |
| Somerst  | 86    | Somerset neighborhood                            |
| Gilbert  | 79    | Gilbert neighborhood                             |
| NridgHt  | 77    | Northridge Heights neighborhood                  |
| Sawyer   | 74    | Sawyer neighborhood                              |
| NWAmes   | 73    | Northwest Ames neighborhood                      |
| SawyerW  | 59    | Sawyer West neighborhood                         |
| BrkSide  | 58    | Brookside neighborhood                           |
| Crawfor  | 51    | Crawford neighborhood                            |
| Mitchel  | 49    | Mitchel neighborhood                             |
| NoRidge  | 41    | Northridge neighborhood                          |
| Timber   | 38    | Timberland neighborhood                          |
| IDOTRR   | 37    | Iowa DOT and Railroad neighborhood               |
| ClearCr  | 28    | Clear Creek neighborhood                         |
| SWISU    | 25    | South & Southwest Iowa State University neighborhood|
| StoneBr  | 25    | Stone Brook neighborhood                         |
| Blmngtn  | 17    | Bloomington neighborhood                         |
| MeadowV  | 17    | Meadow Village neighborhood                      |
| BrDale   | 16    | Briardale neighborhood                           |
| Veenker  | 11    | Veenker neighborhood                             |
| NPkVill  | 9     | Northpark Villa neighborhood                      |
| Blueste  | 2     | Bluestem neighborhood                            |

**13. Number of non-null rows of column Condition1: 1460 / 1460**

| Value  | Count | Explanation                                                  |
|--------|-------|--------------------------------------------------------------|
| Norm   | 1260  | Normal condition, no special proximity to roads/railroads.   |
| Feedr  | 81    | Near feeder road.                                             |
| Artery | 48    | Near arterial street, busy road.                             |
| RRAn   | 26    | Adjacent to railroad with active use noise nuisance.          |
| PosN   | 19    | Near north side of railroad.                                 |
| RRAe   | 11    | Near east side of railroad.                                 |
| PosA   | 8     | Near arterial road.                                          |
| RRNn   | 5     | Near north railroad without noise.                           |
| RRNe   | 2     | Near east railroad without noise.                            |


**14. Number of non-null rows of column Condition2: 1460 / 1460**

| Value  | Count | Explanation                                                  |
|--------|-------|--------------------------------------------------------------|
| Norm   | 1445  | Normal condition, no special proximity.                       |
| Feedr  | 6     | Near feeder road.                                             |
| Artery | 2     | Near arterial street.                                         |
| RRNn   | 2     | Near north railroad without noise.                           |
| PosN   | 2     | Near north railroad.                                          |
| PosA   | 1     | Near arterial road.                                          |
| RRAn   | 1     | Near railroad with active noise.                             |
| RRAe   | 1     | Near east railroad.                                          |


**15. Number of non-null rows of column BldgType: 1460 / 1460**

| Value  | Count | Explanation                                                  |
|--------|-------|--------------------------------------------------------------|
| 1Fam   | 1220  | Single-family detached home.                                 |
| TwnhsE | 114   | Townhouse end unit.                                          |
| Duplex | 52    | Duplex, two-family dwelling.                                |
| Twnhs  | 43    | Townhouse inside unit.                                       |
| 2fmCon | 31    | Two-family conversion or multi-family.                      |


**16. Number of non-null rows of column HouseStyle: 1460 / 1460**

| Value   | Count | Explanation                                                  |
|---------|-------|--------------------------------------------------------------|
| 1Story  | 726   | One-story house.                                             |
| 2Story  | 445   | Two-story house.                                             |
| 1.5Fin  | 154   | One and a half story house with finished attic.             |
| SLvl    | 65    | Split level house.                                           |
| SFoyer  | 37    | Split foyer house.                                           |
| 1.5Unf  | 14    | One and a half story with unfinished attic.                 |
| 2.5Unf  | 11    | Two and a half story unfinished.                            |
| 2.5Fin  | 8     | Two and a half story finished.                              |


**17. Number of non-null rows of column OverallQual: 1460 / 1460**

| Value | Count | Explanation                                                |
|-------|-------|------------------------------------------------------------|
| 5     | 397   | Average quality.                                           |
| 6     | 374   | Above average quality.                                    |
| 7     | 319   | Good quality.                                             |
| 8     | 168   | Very good quality.                                        |
| 4     | 116   | Below average quality.                                    |
| 9     | 43    | Excellent quality.                                       |
| 3     | 20    | Fair quality.                                            |
| 10    | 18    | Very excellent quality (best).                           |
| 2     | 3     | Poor quality.                                            |
| 1     | 2     | Very poor quality (worst).                               |


**18. Number of non-null rows of column OverallCond: 1460 / 1460**

| Value | Count | Explanation                                                |
|-------|-------|------------------------------------------------------------|
| 5     | 821   | Average condition.                                        |
| 6     | 252   | Above average condition.                                 |
| 7     | 205   | Good condition.                                          |
| 8     | 72    | Very good condition.                                     |
| 4     | 57    | Below average condition.                                 |
| 3     | 25    | Fair condition.                                         |
| 9     | 22    | Excellent condition.                                    |
| 2     | 5     | Poor condition.                                         |
| 1     | 1     | Very poor condition.                                    |

**21. Number of non-null rows of column RoofStyle: 1460 / 1460**

| Value   | Count | Explanation                                         |
|---------|-------|---------------------------------------------------|
| Gable   | 1141  | Gable roof, a triangular shape roof common in homes. |
| Hip     | 286   | Hip roof with slopes on all four sides.           |
| Flat    | 13    | Flat roof with minimal or no slope.                |
| Gambrel | 11    | Two-sided roof with two slopes per side (barn style). |
| Mansard | 7     | Four-sided gambrel roof, often with dormers.      |
| Shed    | 2     | Single sloped roof plane.                          |

**22. Number of non-null rows of column RoofMatl: 1460 / 1460**

| Value     | Count | Explanation                                  |
|-----------|-------|----------------------------------------------|
| CompShg   | 1434  | Composite shingles, most common roofing material. |
| Tar&Grv   | 11    | Tar and gravel roofing.                       |
| WdShngl   | 6     | Wood shingles.                                |
| WdShake   | 5     | Wood shakes, split wood roofing.              |
| Metal     | 1     | Metal roofing material.                        |
| Membran   | 1     | Membrane roofing for flat roofs.               |
| Roll      | 1     | Roll roofing, a form of asphalt roofing.      |
| ClyTile   | 1     | Clay tile roofing.                            |

**23. Number of non-null rows of column Exterior1st: 1460 / 1460**

| Value    | Count | Explanation                               |
|----------|-------|-------------------------------------------|
| VinylSd  | 515   | Vinyl siding exterior finish.             |
| HdBoard  | 222   | Hardboard panels.                         |
| MetalSd  | 220   | Metal siding.                            |
| Wd Sdng  | 206   | Wood siding.                             |
| Plywood  | 108   | Plywood exterior sheath.                 |
| CemntBd  | 61    | Cement board siding.                     |
| BrkFace  | 50    | Brick facing.                            |
| WdShing  | 26    | Wood shingles siding.                    |
| Stucco   | 25    | Stucco exterior finish.                  |
| AsbShng  | 20    | Asbestos shingles (rare).                |
| BrkComm  | 2     | Common brick (usually interior or less decorative). |
| Stone    | 2     | Stone siding.                            |
| AsphShn  | 1     | Asphalt shingles.                        |
| ImStucc  | 1     | Imitation stucco.                        |
| CBlock   | 1     | Concrete block.                          |

**24. Number of non-null rows of column Exterior2nd: 1460 / 1460**

| Value    | Count | Explanation                               |
|----------|-------|-------------------------------------------|
| VinylSd  | 504   | Vinyl siding exterior finish.             |
| MetalSd  | 214   | Metal siding.                            |
| HdBoard  | 207   | Hardboard panels.                        |
| Wd Sdng  | 197   | Wood siding.                             |
| Plywood  | 142   | Plywood exterior sheath.                 |
| CmentBd  | 60    | Cement board siding.                     |
| WdShing  | 38    | Wood shingles siding.                    |
| Stucco   | 26    | Stucco exterior finish.                  |
| BrkFace  | 25    | Brick facing.                            |
| AsbShng  | 20    | Asbestos shingles.                       |
| ImStucc  | 10    | Imitation stucco.                        |
| Brk Cmn  | 7     | Common brick.                           |
| Stone    | 5     | Stone siding.                           |
| AsphShn  | 3     | Asphalt shingles.                       |
| Other    | 1     | Other materials.                        |
| CBlock   | 1     | Concrete block.                         |

**25. Number of non-null rows of column MasVnrType: 588 / 1460**

| Value   | Count | Explanation                    |
|---------|-------|--------------------------------|
| BrkFace | 445   | Brick veneer facing.           |
| Stone   | 128   | Stone veneer facing.           |
| BrkCmn  | 15    | Common brick veneer.           |

**27. Number of non-null rows of column ExterQual: 1460 / 1460**

| Value | Count | Explanation                            |
|-------|-------|--------------------------------------|
| TA    | 906   | Typical/Average quality.              |
| Gd    | 488   | Good quality.                        |
| Ex    | 52    | Excellent quality.                   |
| Fa    | 14    | Fair quality.                       |

**28. Number of non-null rows of column ExterCond: 1460 / 1460**

| Value | Count | Explanation                            |
|-------|-------|--------------------------------------|
| TA    | 1282  | Typical/Average condition.            |
| Gd    | 146   | Good condition.                      |
| Fa    | 28    | Fair condition.                     |
| Ex    | 3     | Excellent condition.                |
| Po    | 1     | Poor condition.                    |

**29. Number of non-null rows of column Foundation: 1460 / 1460**

| Value   | Count | Explanation                                |
|---------|-------|--------------------------------------------|
| PConc   | 647   | Poured concrete foundation.                |
| CBlock  | 634   | Concrete block foundation.                  |
| BrkTil  | 146   | Brick and tile foundation.                  |
| Slab    | 24    | Slab foundation.                           |
| Stone   | 6     | Stone foundation.                          |
| Wood    | 3     | Wood foundation, rare or older construction. |

**30. Number of non-null rows of column BsmtQual: 1423 / 1460**

| Value | Count | Explanation                        |
|-------|-------|----------------------------------|
| TA    | 649   | Typical/Average basement quality.|
| Gd    | 618   | Good basement quality.           |
| Ex    | 121   | Excellent basement quality.      |
| Fa    | 35    | Fair basement quality.           |

**31. Number of non-null rows of column BsmtCond: 1423 / 1460**

| Value | Count | Explanation                        |
|-------|-------|----------------------------------|
| TA    | 1311  | Typical/Average basement condition.|
| Gd    | 65    | Good basement condition.          |
| Fa    | 45    | Fair basement condition.          |
| Po    | 2     | Poor basement condition.          |

**32. Number of non-null rows of column BsmtExposure: 1422 / 1460**

| Value | Count | Explanation                               |
|-------|-------|-------------------------------------------|
| No    | 953   | No basement exposure (no walkout).        |
| Av    | 221   | Average exposure.                         |
| Gd    | 134   | Good exposure (walkout or garden level). |
| Mn    | 114   | Minimum exposure.                         |

**33. Number of non-null rows of column BsmtFinType1: 1423 / 1460**

| Value | Count | Explanation                                     |
|-------|-------|-------------------------------------------------|
| Unf   | 430   | Unfinished basement area.                        |
| GLQ   | 418   | Finished basement area - Good living quarters.  |
| ALQ   | 220   | Finished basement area - Average living quarters.|
| BLQ   | 148   | Finished basement area - Below average living quarters.|
| Rec   | 133   | Finished basement area - Recreational room.     |
| LwQ   | 74    | Finished basement area - Low quality living quarters.|

**35. Number of non-null rows of column BsmtFinType2: 1422 / 1460**

| Value | Count | Explanation                                     |
|-------|-------|-------------------------------------------------|
| Unf   | 1256  | Unfinished basement area.                        |
| Rec   | 54    | Finished basement area - Recreational room.     |
| LwQ   | 46    | Finished basement area - Low quality living quarters.|
| BLQ   | 33    | Finished basement area - Below average living quarters.|
| ALQ   | 19    | Finished basement area - Average living quarters.|
| GLQ   | 14    | Finished basement area - Good living quarters.  |

**39. Number of non-null rows of column Heating: 1460 / 1460**

| Value | Count | Explanation                              |
|-------|-------|------------------------------------------|
| GasA  | 1428  | Gas forced warm air furnace.             |
| GasW  | 18    | Gas hot water or steam heat.             |
| Grav  | 7     | Gravity furnace heat.                    |
| Wall  | 4     | Wall furnace heat.                       |
| OthW  | 2     | Other type of heat.                      |
| Floor | 1     | Floor furnace heat.                      |



**40. Number of non-null rows of column HeatingQC: 1460 / 1460**

| Value | Count | Explanation                             |
|-------|-------|-----------------------------------------|
| Ex    | 741   | Excellent heating quality.              |
| TA    | 428   | Typical/average heating quality.        |
| Gd    | 241   | Good heating quality.                    |
| Fa    | 49    | Fair heating quality.                    |
| Po    | 1     | Poor heating quality.                    |


**41. Number of non-null rows of column CentralAir: 1460 / 1460**

| Value | Count | Explanation                            |
|-------|-------|----------------------------------------|
| Y     | 1365  | Yes, central air conditioning.         |
| N     | 95    | No central air conditioning.           |


**42. Number of non-null rows of column Electrical: 1459 / 1460**

| Value | Count | Explanation                                        |
|-------|-------|----------------------------------------------------|
| SBrkr | 1334  | Standard circuit breakers and fuses.               |
| FuseA | 94    | Fuse box - type A.                                 |
| FuseF | 27    | Fuse box - type F.                                 |
| FuseP | 3     | Fuse box - type P with split circuits.             |
| Mix   | 1     | Mixed types of electrical systems.                 |


**45. Number of non-null rows of column LowQualFinSF: 1460 / 1460**

| Value | Count | Explanation                              |
|-------|-------|------------------------------------------|
| 0     | 1434  | No low quality finished square feet.    |
| Others| 26    | Various small amounts of low-quality finished area in square feet. |

**47. Number of non-null rows of column BsmtFullBath: 1460 / 1460**

| Value | Count | Explanation                               |
|-------|-------|-------------------------------------------|
| 0     | 856   | No full bathrooms in the basement.       |
| 1     | 588   | One full bathroom in the basement.       |
| 2     | 15    | Two full bathrooms in the basement.      |
| 3     | 1     | Three full bathrooms in the basement.    |

**48. Number of non-null rows of column BsmtHalfBath: 1460 / 1460**

| Value | Count | Explanation                               |
|-------|-------|-------------------------------------------|
| 0     | 1378  | No half bathrooms in the basement.       |
| 1     | 80    | One half bathroom in the basement.       |
| 2     | 2     | Two half bathrooms in the basement.      |

**49. Number of non-null rows of column FullBath: 1460 / 1460**

| Value | Count | Explanation                               |
|-------|-------|-------------------------------------------|
| 2     | 768   | Two full bathrooms above the basement.   |
| 1     | 650   | One full bathroom above the basement.    |
| 3     | 33    | Three full bathrooms above the basement. |
| 0     | 9     | No full bathrooms above the basement.    |

**50. Number of non-null rows of column HalfBath: 1460 / 1460**

| Value | Count | Explanation                               |
|-------|-------|-------------------------------------------|
| 0     | 913   | No half bathrooms above the basement.    |
| 1     | 535   | One half bathroom above the basement.    |
| 2     | 12    | Two half bathrooms above the basement.   |

**51. Number of non-null rows of column BedroomAbvGr: 1460 / 1460**

| Value | Count | Explanation                               |
|-------|-------|-------------------------------------------|
| 3     | 804   | Three bedrooms above ground.              |
| 2     | 358   | Two bedrooms above ground.                |
| 4     | 213   | Four bedrooms above ground.               |
| 1     | 50    | One bedroom above ground.                 |
| 5     | 21    | Five bedrooms above ground.               |
| 6     | 7     | Six bedrooms above ground.                |
| 0     | 6     | No bedrooms above ground.                 |
| 8     | 1     | Eight bedrooms above ground.              |

**52. Number of non-null rows of column KitchenAbvGr: 1460 / 1460**

| Value | Count | Explanation                               |
|-------|-------|-------------------------------------------|
| 1     | 1392  | One kitchen above ground.                 |
| 2     | 65    | Two kitchens above ground.                |
| 3     | 2     | Three kitchens above ground.              |
| 0     | 1     | No kitchen above ground.                  |

**53. Number of non-null rows of column KitchenQual: 1460 / 1460**

| Value | Count | Explanation                               |
|-------|-------|-------------------------------------------|
| TA    | 735   | Typical/Average kitchen quality.          |
| Gd    | 586   | Good kitchen quality.                      |
| Ex    | 100   | Excellent kitchen quality.                 |
| Fa    | 39    | Fair kitchen quality.                      |

**54. Number of non-null rows of column TotRmsAbvGrd: 1460 / 1460**

| Value | Count | Explanation                                      |
|-------|-------|------------------------------------------------|
| 6     | 402   | Six total rooms above ground (excluding bathrooms). |
| 7     | 329   | Seven total rooms above ground.                 |
| 5     | 275   | Five total rooms above ground.                   |
| 8     | 187   | Eight total rooms above ground.                  |
| 4     | 97    | Four total rooms above ground.                    |
| 9     | 75    | Nine total rooms above ground.                    |
| 10    | 47    | Ten total rooms above ground.                     |
| 11    | 18    | Eleven total rooms above ground.                  |
| 3     | 17    | Three total rooms above ground.                   |
| 12    | 11    | Twelve total rooms above ground.                  |
| 2     | 1     | Two total rooms above ground.                      |
| 14    | 1     | Fourteen total rooms above ground.                |

**55. Number of non-null rows of column Functional: 1460 / 1460**

| Value | Count | Explanation                                          |
|-------|-------|------------------------------------------------------|
| Typ   | 1360  | Typical functionality without significant flaws.    |
| Min2  | 34    | Minor issues, level 2 functionality.                |
| Min1  | 31    | Minor issues, level 1 functionality.                |
| Mod   | 15    | Moderate issues impacting functionality.            |
| Maj1  | 14    | Major issues, level 1 functionality.                |
| Maj2  | 5     | Major issues, level 2 functionality.                |
| Sev   | 1     | Severe issues affecting functionality.              |

**56. Number of non-null rows of column Fireplaces: 1460 / 1460**

| Value | Count | Explanation                          |
|-------|-------|------------------------------------|
| 0     | 690   | No fireplaces in the house.         |
| 1     | 650   | One fireplace present.              |
| 2     | 115   | Two fireplaces present.             |
| 3     | 5     | Three fireplaces present.           |

**57. Number of non-null rows of column FireplaceQu: 770 / 1460**

| Value | Count | Explanation                          |
|-------|-------|------------------------------------|
| Gd    | 380   | Good quality fireplace.             |
| TA    | 313   | Typical/average quality fireplace.  |
| Fa    | 33    | Fair quality fireplace.             |
| Ex    | 24    | Excellent quality fireplace.        |
| Po    | 20    | Poor quality fireplace.             |

**58. Number of non-null rows of column GarageType: 1379 / 1460**

| Value   | Count | Explanation                          |
|---------|-------|------------------------------------|
| Attchd  | 870   | Attached garage.                   |
| Detchd  | 387   | Detached garage.                   |
| BuiltIn | 88    | Built-in garage.                   |
| Basment | 19    | Garage in basement.                |
| CarPort | 9     | Carport (covered open area).      |
| 2Types  | 6     | More than one type of garage.      |

**60. Number of non-null rows of column GarageFinish: 1379 / 1460**

| Value | Count | Explanation                          |
|-------|-------|------------------------------------|
| Unf   | 605   | Unfinished garage.                 |
| RFn   | 422   | Garage with rough finish.          |
| Fin   | 352   | Finished garage.                   |

**61. Number of non-null rows of column GarageCars: 1460 / 1460**

| Value | Count | Explanation                          |
|-------|-------|------------------------------------|
| 2     | 824   | Garage can hold two cars.           |
| 1     | 369   | Garage can hold one car.            |
| 3     | 181   | Garage can hold three cars.         |
| 0     | 81    | No garage.                         |
| 4     | 5     | Garage can hold four cars.          |

**63. Number of non-null rows of column GarageQual: 1379 / 1460**

| Value | Count | Explanation                          |
|-------|-------|------------------------------------|
| TA    | 1311  | Typical/average garage quality.     |
| Fa    | 48    | Fair garage quality.                |
| Gd    | 14    | Good garage quality.                |
| Ex    | 3     | Excellent garage quality.           |
| Po    | 3     | Poor garage quality.                |

**64. Number of non-null rows of column GarageCond: 1379 / 1460**

| Value | Count | Explanation                          |
|-------|-------|------------------------------------|
| TA    | 1326  | Typical/average garage condition.   |
| Fa    | 35    | Fair garage condition.              |
| Gd    | 9     | Good garage condition.              |
| Po    | 7     | Poor garage condition.              |
| Ex    | 2     | Excellent garage condition.          |

**65. Number of non-null rows of column PavedDrive: 1460 / 1460**

| Value | Count | Explanation                          |
|-------|-------|------------------------------------|
| Y     | 1340  | Paved driveway.                    |
| N     | 90    | No driveway.                      |
| P     | 30    | Partially paved driveway.          |

**69. Number of non-null rows of column 3SsnPorch: 1460 / 1460**

| Value | Count | Explanation                          |
|-------|-------|------------------------------------|
| 0     | 1436  | No three-season porch area.         |
| Others| 24    | Square footage of three-season porch area (various small sizes). |

**71. Number of non-null rows of column PoolArea: 1460 / 1460**

| Value | Count | Explanation                          |
|-------|-------|------------------------------------|
| 0     | 1453  | No pool area.                      |
| Others| 7     | Square footage of pool area (various sizes). |

**72. Number of non-null rows of column PoolQC: 7 / 1460**

| Value | Count | Explanation                          |
|-------|-------|------------------------------------|
| Gd    | 3     | Good pool quality.                  |
| Ex    | 2     | Excellent pool quality.             |
| Fa    | 2     | Fair pool quality.                  |

**73. Number of non-null rows of column Fence: 281 / 1460**

| Value | Count | Explanation                          |
|-------|-------|------------------------------------|
| MnPrv | 157   | Privacy fence (wood).               |
| GdPrv | 59    | Good privacy fence.                 |
| GdWo  | 54    | Good wood fence.                   |
| MnWw  | 11    | Minimum wood/wire fence.           |

**74. Number of non-null rows of column MiscFeature: 54 / 1460**

| Value | Count | Explanation                          |
|-------|-------|------------------------------------|
| Shed  | 49    | Shed located on property.           |
| Gar2  | 2     | Second garage.                     |
| Othr  | 2     | Other miscellaneous feature.       |
| TenC  | 1     | Tennis court on property.           |

**75. Number of non-null rows of column MiscVal: 1460 / 1460**

| Value | Count | Explanation                          |
|-------|-------|------------------------------------|
| 0     | 1408  | No miscellaneous value.             |
| Others| 52    | Monetary values of miscellaneous features like sheds, garages, courts, etc. |


**76. Number of non-null rows of column MoSold: 1460 / 1460**

| Value | Count | Explanation                  |
|-------|-------|------------------------------|
| 6     | 253   | Month of sale: June.          |
| 7     | 234   | Month of sale: July.          |
| 5     | 204   | Month of sale: May.           |
| 4     | 141   | Month of sale: April.         |
| 8     | 122   | Month of sale: August.        |
| 3     | 106   | Month of sale: March.         |
| 10    | 89    | Month of sale: October.       |
| 11    | 79    | Month of sale: November.      |
| 9     | 63    | Month of sale: September.     |
| 12    | 59    | Month of sale: December.      |
| 1     | 58    | Month of sale: January.       |
| 2     | 52    | Month of sale: February.      |

**77. Number of non-null rows of column YrSold: 1460 / 1460**

| Value | Count | Explanation                  |
|-------|-------|------------------------------|
| 2009  | 338   | Year of sale: 2009            |
| 2007  | 329   | Year of sale: 2007            |
| 2006  | 314   | Year of sale: 2006            |
| 2008  | 304   | Year of sale: 2008            |
| 2010  | 175   | Year of sale: 2010            |

**78. Number of non-null rows of column SaleType: 1460 / 1460**

| Value | Count | Explanation                                                  |
|-------|-------|--------------------------------------------------------------|
| WD    | 1267  | Warranty deed - standard sale type.                         |
| New   | 122   | New construction sale.                                       |
| COD   | 43    | Cash on delivery/close sale.                                |
| ConLD | 9     | Contract sale - land development.                            |
| ConLI | 5     | Contract sale - land improvement.                            |
| ConLw | 5     | Contract sale - land warranty.                               |
| CWD   | 4     | Contract warranty deed.                                      |
| Oth   | 3     | Other sale types not listed explicitly.                      |
| Con   | 2     | Contract sale.                                               |

**79. Number of non-null rows of column SaleCondition: 1460 / 1460**

| Value     | Count | Explanation                                              |
|-----------|-------|----------------------------------------------------------|
| Normal    | 1198  | Normal sale.                                            |
| Partial   | 125   | Partial sale (e.g., foreclosure or short sale).          |
| Abnorml   | 101   | Abnormal sale (e.g., foreclosure, estate sale).          |
| Family    | 20    | Sale between family members.                            |
| Alloca    | 12    | Allocation sale.                                        |
| AdjLand   | 4     | Land adjacency sale.                                    |


## Columns with more than 50 unique values
|Column         |  Description                                                                 
|---------------|------------------------------------------------------------------------------|
|LotFrontage    |  Linear feet of street connected to the property (length of the frontage).   |
|LotArea        |  Total size of the lot in square feet.                                       |
|YearBuilt      |  Year the house was originally constructed.                                  |
|YearRemodAdd   |  Year the house was remodeled or added to.                                   |
|MasVnrArea     |  Area (in square feet) covered by masonry veneer (decorative stone or brick).|
|BsmtFinSF1     |  Area of finished basement Type 1 in square feet.                            |
|BsmtFinSF2     |  Area of finished basement Type 2 in square feet.                            |
|BsmtUnfSF      |  Area of unfinished basement in square feet.                                 |
|TotalBsmtSF    |  Total basement area in square feet (finished + unfinished).                 |
|1stFlrSF       |  First floor living area in square feet.                                     |
|2ndFlrSF       |  Second floor living area in square feet.                                    |
|GrLivArea      |  Above grade (ground level) living area in square feet, excluding basement.  |
|GarageYrBlt    |  Year the garage was built.                                                  |
|GarageArea     |  Garage size in square feet.                                                 |
|WoodDeckSF     |  Wood deck area in square feet.                                              |
|OpenPorchSF    |  Open porch area in square feet.                                             |
|EnclosedPorch  |  Enclosed porch area in square feet (e.g., sunroom).                         |
|ScreenPorch    |  Screened porch area in square feet.                                         |
|SalePrice      |  Sale price of the property (target variable for prediction).                |