import os
import torch
import numpy as np
import SimpleITK as sitk
from skimage.measure import regionprops
from torch_geometric.utils import coalesce
from pathlib import Path
from glob import glob
from torch_geometric.data import Data
from utils import *
mapping_d={
	'GTV':0,
	'BrainStem':1,
	'Esophagus':2,
	'Larynx':3,
	'Mandible':4,
	'fixed_Parotid':5, #Left:5, Right:6
	'RetroPharynx':7,
	'fixed_Submandibular':8, #Left:8, Right:9
}
roi_name_key={
    'GTV': 'gtv',
    'BrainStem': 'brainstem',
    'Esophagus': 'esophagus',
    'Larynx': 'larynx',
    'Mandible': 'mandible',
    'Right_Parotid': 'rightparotid',
    'Left_Parotid': 'leftparotid',
    'RetroPharynx': 'retropharynx',
    'Left_Submandibular': 'leftsubmandibular',
    'Right_Submandibular': 'rightsubmandibular'
    }
roi_list=[
    'gtv',
    'larynx',
    'retropharynx',
    'leftparotid',
    'rightparotid',
    'brainstem',
    'esophagus',
    'leftsubmandibular',
    'rightsubmandibular',
    'mandible',]


def make_subgraph_data(inpath,y=None):
  """
  A function to create graph data for each of the organs studied.
  inpath is a directory containing a patient CT as well as several mask files in .nii.gz format
  files are named {Patient_ID}__{Image/mask_type}.nii.gz


  y is the patient outcome


  """


  image_path=glob(os.path.join(inpath,'*__CT.nii.gz'))[0]
  gtv_path=glob(os.path.join(inpath,'*__mask.nii.gz'))[0] #This holds the GTV (the tumor mask)

  label_dict={}
  for file_name in os.listdir(inpath):
    patient=file_name.split('__')[0]
    struct=file_name.split('__')[-1].split('.')[0]
    if struct in mapping_d.keys():
      if struct=='fixed_Submandibular':
        label_dict['Submandibular']=os.path.join(inpath,file_name)  
      elif struct=='fixed_Parotid':
        label_dict['Parotid']=os.path.join(inpath,file_name)
      else:
        label_dict[struct]=os.path.join(inpath,file_name)
  patient=Path(image_path).name.split('__')[0] #Just get the patient ID
  image=sitk.ReadImage(image_path)
  image_array=sitk.GetArrayFromImage(image)
  label_dict['GTV']=gtv_path 
  Graph_List={}
  for struct_name,file_path in label_dict.items():
    label_array = sitk.GetArrayFromImage(sitk.ReadImage(file_path)).astype(int) #Read in the mask for a structure
    regions = regionprops(label_array) #Should only be 1 or 2 labels in each. Masks with 2 labels are right, left
    for desired_region in regions:
        curr_mask = label_array == desired_region.label 

        curr_image_array=image_array.copy().astype(int) 
        curr_image_array[curr_mask==0]=0 #mask out everything but the desired organ
        curr_image_array=(curr_image_array-curr_image_array.min())/(curr_image_array.max()-curr_image_array.min()) #min max normalization of voxel values
        current_graph = Data()
        centroid = np.round(desired_region.centroid).astype(int) #Get centroid for use as graph feature
        indices = np.nonzero(curr_mask) 
        num_nodes = len(indices[0]) 

        # Node features
        node_map={}
        node_features=[]
        node_index=0
        for i, j, k in zip(indices[0], indices[1], indices[2]):
          node_features.append((curr_image_array[i,j,k],i,j,k)) #intensity and position as node features
          node_map[(i,j,k)]=node_index 
          node_index+=1

        # Edge indices
        edge_index = []
        for i, j, k in zip(indices[0], indices[1], indices[2]): 
            for dx in range(-1, 2): #search nearest neighbors
                if i+dx>=curr_mask.shape[0]: #if outside the mask, continue
                  continue
                elif i+dx<0: 
                  continue
                for dy in range(-1, 2):
                    if j+dy>=curr_mask.shape[1]:
                      continue
                    elif j+dy<0:
                      continue
                    for dz in range(-1, 2):
                        if k+dz>=curr_mask.shape[2]:
                          continue
                        elif k+dz<0:
                          continue
                        if dx == 0 and dy == 0 and dz == 0:
                            continue  # Skip the central point itself
                        if curr_mask[i + dx, j + dy, k + dz]: #as long as there is a value (remember we masked out non-organ voxels)
                            edge_index.append((node_map[i,j,k],node_map[i + dx, j + dy, k + dz])) 

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous() #Transpose important for torch geometric format
        if edge_index.numel():
          edge_index = coalesce(edge_index) #Get rid of any repeated edges
        current_graph.x = torch.tensor(node_features,dtype=torch.float)

        current_graph.edge_index = edge_index
        if struct_name in ('Submandibular','Parotid'):
          if desired_region.label==1:
            name='Left_'+struct_name
          elif desired_region.label==2:
            name='Right_'+struct_name
        else:
          name=struct_name
        current_graph.organ=name
        Graph_List[name]=(current_graph,centroid)

  argument_dict={} #Creating a kwargs dict for the AnatomicalData class
  for struct,(graph,cent) in Graph_List.items():
    arg_struct_name=roi_name_key[struct] #Just translating file names to be the same as what I use later
    argument_dict[arg_struct_name]=graph.x 
    argument_dict['edge_index_'+arg_struct_name]=graph.edge_index
  argument_dict['patient']=patient
  argument_dict['centroid']={roi_name_key[r]:c[1] for r,c in Graph_List.items()}
  argument_dict['y']=y
  for roi in roi_list:
    if roi not in argument_dict.keys():
      if roi=='gtv': #Don't include patients with no GTV (tumor) contours
        raise(Exception('MISSING GTV'))
      argument_dict[roi]=torch.zeros((1,4)) #Zero feature array for missing structure. 
      argument_dict['edge_index_'+roi]=torch.tensor([],dtype=torch.long) #No edges for dummy struct, but you could add dummy edges
      argument_dict['centroid'][roi]=np.array([0,0,0]) #This is how we'll know an ROI is missing later. 
      #In principle, there should never be an ROI with an actual centroid of (0,0,0). Not anatomically feasible.
  subgraph=AnatomicalData(**argument_dict)
  return subgraph