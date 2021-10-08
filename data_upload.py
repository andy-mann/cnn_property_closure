from cordra import CordraClient, Dataset
import numpy as np
from io import BytesIO
import h5py
import os

cwd = os.getcwd()

train_r = os.path.join(cwd, '..', '..', '..', 'Elastic_Localization_Data', 'cr_50_full', '31_c50_train_responses.h5')
train_m = os.path.join(cwd, '..', '..', '..', 'Elastic_Localization_Data', 'cr_50_full', '31_c50_train_micros.h5')

train_m = h5py.File(train_m)
train_r = h5py.File(train_r)


m = train_m['micros']
train_stress = train_r['stress']
train_strain = train_r['strain']

avg_stress = np.sum(train_stress, axis=1) / 31**3
avg_strain = np.sum(train_strain, axis=1) / 31**3

dp = len(avg_stress)

host = 'https://dev.materialhub.org/'
creds = 'credentials.json'

materialhub = CordraClient(host=host, credentials_file=creds, verify=False)

dataset = Dataset(client=materialhub)

ds = dataset.add('Dataset', properties={'name':'2_phase_elastic_CR50'})

#defining terms
#TODO add definitions and more specifics
YM_term = dataset.add('DefinedTerm', properties={'name':'Young\'s Modulus'})
PR_term = dataset.add('DefinedTerm', properties={'name':'Poisson Ratio'})
effective_property = dataset.add('DefinedTerm', properties={'name':'Effective Property'})
broad_mat_term = dataset.add('DefinedTerm', properties={'name':'Two Phase Material'})
stress_field = dataset.add('DefinedTerm', properties={'name':'Stress_Field', 'description':'An array of stress states in each voxel of your domain'})
structure_term = dataset.add('DefinedTerm', properties={'name':'microstructure', 'description':'synthetically generated 2 phase eigen-structure'})
strain_field = dataset.add('DefinedTerm', properties={'name':'Strain_Field'})
abaqus_term = dataset.add('DefinedTerm', properties={'name':'abaqus', 'description':'Finite Element Solver'})
python_term = dataset.add('DefinedTerm', properties={'name':'python', 'description':'programming language'})


#defining the material properties by using the DF, values, and units
m1_ym = dataset.add('MaterialProperty', properties={'PropertyID':YM_term, 'name':YM_term.properties['name'], 'value':'120'})
m1_pr = dataset.add('MaterialProperty', properties={'PropertyID':PR_term, 'name':PR_term.properties['name'], 'value':'0.3'})
                                                                          
m2_ym = dataset.add('MaterialProperty', properties={'PropertyID':YM_term, 'name':YM_term.properties['name'], 'value':'6,000'})
m2_pr = dataset.add('MaterialProperty', properties={'PropertyID':PR_term, 'name':PR_term.properties['name'], 'value':'0.3'})

mat1 = dataset.add('Material', properties={'name':'Low Stiffness Material', 'about':[m1_ym.properties['@id'], m1_pr.properties['@id']]})
mat2 = dataset.add('Material', properties={'name':'High Stiffness Material', 'about':[m2_ym.properties['@id'], m2_pr.properties['@id']]})

broad_material = dataset.add('Material', properties={'PropertyID':broad_mat_term,'name':'2 Phase Material', 'about':[mat1.properties['@id'], mat2.properties['@id']]})


#define python
python = dataset.add('SoftwareApplication', properties={'propertyID':python_term})
#define abaqus
abaqus = dataset.add('SoftwareApplication', properties={'propertyID':abaqus_term})


for i in range(dp):
    val = avg_stress / avg_strain
    eff_p = (avg_stress[i] / avg_strain[i])[0].round()
    #TODO check if the property has been created or not

    narrow_mat_term = dataset.add('DefinedTerm', properties={'name':f'Two Phase - Effective stiffness: {val}', 'broader':[broad_mat_term.properties['@id']]})
    eff_prop = dataset.add('MaterialProperty', properties={'PropertyID':effective_property, 'name':effective_property.properties['name']+f'={val}', 'value':f'{val}'})

                                                        
    narrow_material = dataset.add('Material', properties={'PropertyID':narrow_mat_term, 'name':narrow_mat_term.properties['name'], 'about':[mat1.properties['@id'], mat2.properties['@id'], eff_prop.properties['@id']]})

    structure = dataset.add('MaterialStructure', properties={'propertyID':structure_term,'name':'test structure','material':[broad_material.properties['@id'], narrow_material.properties['@id']]}, payloads={'structure':BytesIO(m[i]).getvalue()})


    stress_result = dataset.add('MaterialProperty', properties={'propertpyID':stress_field, 'material':[broad_material.properties['@id'], narrow_material.properties['@id']]}, payloads={'value':BytesIO(train_stress[i]).getvalue()})
    strain_result = dataset.add('MaterialProperty', properties={'propertpyID':strain_field, 'material':[broad_material.properties['@id'], narrow_material.properties['@id']]}, payloads={'value':BytesIO(train_strain[i]).getvalue()})


    fem = dataset.add('Action', properties={'agent':[abaqus.properties['@id']], 'object':[structure.properties['@id']], 'result':[stress_result.properties['@id'], strain_result.properties['@id']]})


    eff_stiff = dataset.add('Action', properties={'agent':[python.properties['@id']], 'object':[stress_result.properties['@id'], strain_result.properties['@id']], 'result':[narrow_material.properties['@id']] })



