from torch_geometric.data import Data
import pickle
from torch_geometric.data import Dataset
import os
class PickleDataset(Dataset):
  """
  A class for data loading using pickle dictionaries for graphs.
  """
  def __init__(self, overall_path, patient_names):
    self.overall_path=overall_path
    self.patient_names=patient_names
    super().__init__()
  def len(self):
    return len(self.patient_names)

  def get(self, idx):
    patient=self.patient_names[idx]
    inpath=os.path.join(self.overall_path,patient)
    data=load_pkl(inpath)[patient.split('.')[0]]
    return data


def are_items_increasing(lst,num_last=5):
    #used for early stopping in val loop.
    last_items = lst[-num_last:]
    return last_items == sorted(last_items) 


class AnatomicalData(Data):
    """
    Class for customized 10 subgraph data object in torch geometric. 
    """
    def __inc__(self, key, value, *args, **kwargs):

        ref_dict={
            'edge_index_gtv':self.gtv.size(0),
            'edge_index_larynx':self.larynx.size(0),
            'edge_index_retropharynx':self.retropharynx.size(0),
            'edge_index_leftparotid':self.leftparotid.size(0),
            'edge_index_rightparotid':self.rightparotid.size(0),
            'edge_index_brainstem':self.brainstem.size(0),
            'edge_index_esophagus':self.esophagus.size(0),
            'edge_index_leftsubmandibular':self.leftsubmandibular.size(0),
            'edge_index_rightsubmandibular':self.rightsubmandibular.size(0),
            'edge_index_mandible':self.mandible.size(0),

        }

        if key in ref_dict.keys():
            return ref_dict[key]

        return super().__inc__(key, value, *args, **kwargs)


def save_pkl(pkl_dict,outPath):

  filehandler = open(outPath, 'wb')
  pickle.dump(pkl_dict, filehandler)
  filehandler.close()
  return

def load_pkl(pkl_path):

  filehandler = open(pkl_path, 'rb')
  result_dict=pickle.load(filehandler)
  filehandler.close()
  return result_dict


