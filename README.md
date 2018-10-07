# Mapping

To evaluate models of the brain, we need to find the mapping between the units in the model and the neurons in the data.

Unit_test.py: main code which calls 
- DataMapModel class with the data parameters (ni= number of images, 
nf = number of features, and nt = number of trials) 
- MappingUnitTest class with the data, map and the desired preprocessing applied to the model(D = data,
 Dmu = true mean of data, A = Map, PCA_ncomponents_list, explained_var_ratio_list)
  MappingUnitTest.get_model builds the model based on tha provided parameters (data, map and 
  preprocessing)
  MappingUnitTest.get_mappings_unit_test performs the mapping 