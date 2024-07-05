import torch
from torch_geometric.nn.conv import GATConv, GCNConv
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn.norm import LayerNorm
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import Batch
from torch_geometric.data import Data
import numpy as np
from torch_geometric.nn.pool import TopKPooling
from collections import defaultdict

class HoGNet(torch.nn.Module):
  def __init__(
      self,
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
        'mandible',
      ],
      inner_settings={
        'layers':1,
        'hidden_dim':8,
      },
      outer_settings={
        'layers':2,
        'hidden_dim':16,
        'heads':4
      },
      dropout=0,
      device=torch.device("cuda"),
      norm=1
  ):
    super(HoGNet,self).__init__()
    torch.manual_seed(1)
    self.norm=norm
    self.roi_list=roi_list
    self.inner_layers=inner_settings['layers']
    self.inner_hidden_dim=inner_settings['hidden_dim']

    self.outer_layers=outer_settings['layers']
    self.outer_hidden_dim=outer_settings['hidden_dim']
    self.outer_heads=outer_settings['heads']


    self.dropout=dropout
    self.device=device
    self.inner_gat_layer_lists = torch.nn.ModuleList()
    #Create GCNs for each OAR studied
    for i in range(len(roi_list)):

      curr_layer_list=torch.nn.ModuleList()
      curr_layer_list.append(GCNConv(4, self.inner_hidden_dim))
      if self.norm:
        curr_layer_list.append(LayerNorm(self.inner_hidden_dim))
      curr_layer_list.append(TopKPooling(self.inner_hidden_dim))
      if self.inner_layers>1:
        for _ in range(self.inner_layers - 1):

          curr_layer_list.append(GCNConv(self.inner_hidden_dim, self.inner_hidden_dim))
          if self.norm:
            curr_layer_list.append(LayerNorm(self.inner_hidden_dim))
          curr_layer_list.append(TopKPooling(self.inner_hidden_dim))

      self.inner_gat_layer_lists.append(curr_layer_list)

    self.outer_model=torch.nn.ModuleList()
    self.outer_model.append(GATConv((self.inner_layers+1)*self.inner_hidden_dim+3, self.outer_hidden_dim, heads=self.outer_heads, dropout=self.dropout))
    if self.norm:
      self.outer_model.append(LayerNorm(self.outer_hidden_dim*self.outer_heads))

    if self.outer_layers>1:
      for _ in range(self.outer_layers - 2):

        self.outer_model.append(GATConv(self.outer_hidden_dim * self.outer_heads, self.outer_hidden_dim, heads=self.outer_heads, dropout=self.dropout))
        if self.norm:
          self.outer_model.append(LayerNorm(self.outer_hidden_dim*self.outer_heads))
      self.outer_model.append(GATConv(self.outer_hidden_dim * self.outer_heads, self.outer_hidden_dim, heads=1, dropout=self.dropout))
      if self.norm:
        self.outer_model.append(LayerNorm(self.outer_hidden_dim))

    if self.outer_layers==1:
      self.lin=Linear(self.outer_hidden_dim*self.outer_heads,2)
    else:
      self.lin=Linear(self.outer_hidden_dim,2)


  def forward(self,data):
    roi_feature_embeddings={} #Will become node features for SuperGraph
    for roi, model_layers in zip(self.roi_list,self.inner_gat_layer_lists):
      residuals=[]
      roi_x,roi_edge_index,roi_batch=data[roi].double(),data['edge_index_'+roi],data[roi+'_batch']
      for individual_layer in model_layers:
        if isinstance(individual_layer,TopKPooling):  
          roi_x, roi_edge_index, _, roi_batch,_,_=individual_layer(roi_x,roi_edge_index,batch=roi_batch)
          if not self.norm: #If we aren't norming, get the residuals after the TopK Pooling
            residuals.append(global_mean_pool(roi_x,roi_batch))
        elif isinstance(individual_layer,LayerNorm): #If we are norming, then get residuals after norm
          roi_x=individual_layer(roi_x,roi_batch)
          residuals.append(global_mean_pool(roi_x,roi_batch))
        elif isinstance(individual_layer,GCNConv): 
          roi_x=individual_layer(roi_x,roi_edge_index)
          roi_x=F.relu(roi_x)
        else:
          raise(Exception("Unknown Layer Type"))
      feature_vector=global_mean_pool(roi_x,roi_batch) 
      roi_centroids=torch.tensor(np.array(data.centroid[roi]),dtype=torch.double).to(self.device) #Will be zeros if ROI not present in data to begin with
      feature_vector=torch.cat((feature_vector,roi_centroids),axis=1)
      for res in residuals:
        feature_vector=torch.cat((res,feature_vector),axis=1)
      roi_feature_embeddings[roi]=feature_vector #Cat all residuals at start of input to SuperGraph
    super_graph_list=[]
    super_graph_assignment_ref_dict={}
    for i in range(len(data)):
      curr_graph=Data()
      node_features=[]
      assignment_dict={}
      assignment_counter=0
      for r in self.roi_list:
        feature_vector=roi_feature_embeddings[r][i]
        if not torch.all(feature_vector[-3::]==torch.zeros((1,3)).to(self.device)): #Do not add nodes to SuperGraph for OARs not present

          node_features.append(feature_vector)
          assignment_dict[assignment_counter]=r
          assignment_counter+=1
        # else:
        #   print(i,'dropped ROI')
      curr_graph.x=torch.stack(node_features)
      #Below creates edges between GTV node and all OAR nodes in SuperGraph
      super_edges=torch.tensor(
                              [
                                [0 for ii in range(len(node_features)-1)],
                                [jj for jj in range(1,len(node_features))]
                              ],
                              dtype=torch.long
                              ).to(self.device)
      curr_graph.edge_index=super_edges
      super_graph_assignment_ref_dict[i]=assignment_dict
      curr_graph['ID']=i
      super_graph_list.append(curr_graph)

    super_batch=Batch.from_data_list(super_graph_list)

    super_x, super_edge_index, super_batch_index = super_batch.x,super_batch.edge_index,super_batch.batch
    for gat_layer in self.outer_model:
      if isinstance(gat_layer,LayerNorm):
        super_x = gat_layer(super_x, super_batch_index)
      else:
        super_x = gat_layer(super_x, super_edge_index)
        super_x = F.elu(super_x)
    # super_x=torch.stack([super_x[super_batch_index==i][0] for i in range(len(super_batch))])
    super_x = global_mean_pool(super_x,super_batch_index)
    super_x = F.dropout(super_x, p=0.5, training=self.training)
    super_x = self.lin(super_x)
    return super_x