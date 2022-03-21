import torch
from  torch import nn
from torch.nn import RNNCell
from torch.nn.functional import one_hot
import math
from torch.nn import functional as F


## This custom class of Linear, enables to initialize the weights of the layer to belong to a normal distribution ##

class NormalLinear(nn.Linear):
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.normal_(0, stdv)
        if self.bias is not None:
            self.bias.data.normal_(0, stdv)

## This function enables to create the dynamic embedding of each node ##
      

class RODIE(torch.nn.Module):

    def __init__(self,embedding_dim,data,device,activation_rnn="tanh",MLP_h_dim=50,option="user_state"):
        super(RODIE, self).__init__()
        self.option = option
        self.embedding_dim = embedding_dim
        self.activation_rnn = activation_rnn
        self.data = data
        self.MLP_h_dim = MLP_h_dim  # The dimension of the hidden layer of the MLP used for the classification of Users 
        # Select features of the data
        self.features = self.data[:,8:]
        self.dim_features = self.features.shape[1]
        # Number of users and number of items
        num_users = len(torch.unique(data[:,0]))

        num_items = len(torch.unique(data[:,1]))


        print("Number of users of {} \n Number of items {} \n".format(num_users,num_items))
        print("Dataset size {}".format(list(self.data.size())))
        # Initialize static  embeddings
        self.static_users_embedding = one_hot(torch.arange(0,num_users))
        self.static_items_embedding = one_hot(torch.arange(0,num_items+1))
        static_user_embedding_dim = self.static_users_embedding.shape[1]
        static_item_embedding_dim = self.static_items_embedding.shape[1]
        print("Initialisation of static embedding... Done !")
        print("Static Embedding shape : Users {}, \t Items {}".format(list(self.static_users_embedding.size()),list(self.static_items_embedding.size())))

        # Initialize dynamic  embeddings
        # In JODIE official implementation, authors decided to attribute the SAME initial dynamic embedding 
        



        input_rnn_user_dim =  self.embedding_dim + self.dim_features + 1

        input_rnn_item_dim =  self.embedding_dim + self.dim_features + 1

        self.item_rnn = RNNCell(input_rnn_user_dim, self.embedding_dim, nonlinearity = self.activation_rnn)


        self.user_rnn = RNNCell(input_rnn_item_dim,self.embedding_dim, nonlinearity = self.activation_rnn)

        print("Initialisation of rnn's with {} activation function... Done !".format(self.activation_rnn))

        # Projection layer -> projection operation   
        self.projection_layer = NormalLinear(1,self.embedding_dim, bias=False)
        # Predict next item embedding layer
        self.predictItem_layer = torch.nn.Sequential(
            nn.Linear(static_item_embedding_dim + static_user_embedding_dim  + 2*self.embedding_dim, static_item_embedding_dim + self.embedding_dim),
          #  torch.nn.Tanh()
        )

        self.predictStateUser_MLP = torch.nn.Sequential(
            nn.Linear(self.embedding_dim,self.MLP_h_dim),
            torch.nn.ReLU(),
            nn.Linear(self.MLP_h_dim,2),
            #torch.nn.Softmax(dim=1)
            )
        print("Initialisation of MLP... Done !")


    ######## Predicting next item embedding  ########
    def update_item_rnn(self,
                          dynamic_item_embedding, # at t-1
                          dynamic_user_embedding,# at t-1
                          features,
                          delta_i,
):

      concat_input = torch.concat([
                                  dynamic_user_embedding,
                                  delta_i.reshape(-1,1),
                                  features,
      ],axis=1)
      return F.normalize(self.item_rnn(concat_input,dynamic_item_embedding))



    ######## Predicting next user embedding  ########
    def update_user_rnn(self,
                        dynamic_user_embedding, # at t-1
                        dynamic_item_embedding,# at t-1
                        features,
                        delta_u):
      concat_input = torch.concat([
                                  dynamic_item_embedding,
                                  delta_u.reshape(-1,1),
                                   features],
                                  axis=1)
      return F.normalize(self.user_rnn(concat_input,dynamic_user_embedding))

    

    ######## Projecting the embedding the new dynamic embedding of the user at a future time  ########
    def projection_operation(self,
                            dynamic_user_embedding,
                            delta_u):
        u_projection =  dynamic_user_embedding * (1 + self.projection_layer(delta_u.reshape(-1,1)))

        return u_projection
        
    ######## Predicting next potential item, the specific user will interact with  ########
    
    def predict_item_embedding(self,
        u_projection,
        u_static,
        i_dynamic,
        i_static
        ):
        concatenated_input = torch.concat([u_projection,i_dynamic,u_static,i_static],axis=1)
        j_tilde = self.predictItem_layer(concatenated_input)
        return j_tilde

    ######## Predicting next user state  ########

    def predict_user_state(self,dynamic_user_embedding):
        u_state = self.predictStateUser_MLP(dynamic_user_embedding)

        return u_state

    def forward(self,
                actual_item_embedding,
                actual_user_embedding,
                u_static,
                i_static,
                f,
                delta_u,
                delta_i,
                next_state_label,
                past_item_dynamic_embedding,
                past_item_static_embedding              
                ):
      # Projection of the user
      
      projected_user_embedding = self.projection_operation(actual_user_embedding,delta_u)
      # Predict next item
      j_tilde = self.predict_item_embedding(
        projected_user_embedding,
        past_item_dynamic_embedding,
        u_static,
        past_item_static_embedding)
      # The real next item embedding j_true, is the concatenation of the static and dynamic embedding of the next item 
      j_true = torch.concat([actual_item_embedding,i_static],axis=1).detach()


      ######## New  Dynamic Embeddings ########
      # New dynamic embedding of the user
      future_user_embedding= self.update_user_rnn(actual_user_embedding,actual_item_embedding,f,delta_u)
      # New dynamic embedding of the item
      future_item_embedding= self.update_item_rnn(actual_item_embedding,actual_user_embedding,f,delta_i)     

      # Prediction of next state of the user using an MLP at the end
      U_pred_state = self.predict_user_state(actual_user_embedding)

      return future_user_embedding, future_item_embedding, U_pred_state, j_tilde, j_true
