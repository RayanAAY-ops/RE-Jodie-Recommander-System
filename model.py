import torch
from  torch import nn
from torch.nn import RNNCell
from torch.nn.functional import one_hot
import math
from torch.nn import MSELoss, HuberLoss,L1Loss,CrossEntropyLoss
from torch.nn import functional as F

def regularizer(actual_user_embedding,future_user_embedding,lambda_u,
                               actual_item_embedding,future_item_embedding,lambda_i
                               ):
    u_regularization_loss =  MSELoss()(actual_user_embedding,future_user_embedding)
    i_regularization_loss =  MSELoss()(actual_item_embedding,future_item_embedding)
    return lambda_u* u_regularization_loss + lambda_i* i_regularization_loss 


## This custom class of Linear, enables to initialize the weights of the layer to belong to a normal distribution ##

class NormalLinear(nn.Linear):
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.normal_(0, stdv)
        if self.bias is not None:
            self.bias.data.normal_(0, stdv)

## This function enables to create the dynamic embedding of each node ##
def dynamic_embedding(data,embedding_dim):
        num_users = len(torch.unique(data[:,0]))
        num_items = len(torch.unique(data[:,1]))
        dynamic_users_embedding = F.normalize(torch.randn(num_users,embedding_dim))
        dynamic_items_embedding = F.normalize(torch.randn(num_items,embedding_dim))
        print("Initialisation of dynamic embedding... Done !")
        print("Dynamic Embedding shape : Users {}, \t Items {}".format(list(dynamic_users_embedding.size()),list(dynamic_items_embedding.size())))

        return dynamic_users_embedding,dynamic_items_embedding
        

class RODIE(torch.nn.Module):

    def __init__(self,embedding_dim,data,activation_rnn="tanh",MLP_h_dim=50,option="user_state"):
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
        self.static_items_embedding = one_hot(torch.arange(0,num_items))
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

        print("Initialisation of rnn's... Done !")

        # Projection layer -> projection operation   
        self.projection_layer = NormalLinear(1,self.embedding_dim, bias=False)
        # Predict next item embedding layer
        self.predictItem_layer = nn.Linear(static_item_embedding_dim + static_user_embedding_dim  + 2*self.embedding_dim, static_item_embedding_dim + self.embedding_dim, bias=True)

        self.predictStateUser_MLP = torch.nn.Sequential(
            nn.Linear(self.embedding_dim,self.MLP_h_dim),
            torch.nn.ReLU(),
            nn.Linear(self.MLP_h_dim,2),
            torch.nn.Softmax(dim=1)
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
                                  features,
                                  delta_i.reshape(-1,1)
      ],
                                  axis=1)
      
      return F.normalize(self.item_rnn(concat_input,dynamic_item_embedding))



    ######## Predicting next user embedding  ########
    def update_user_rnn(self,
                        dynamic_user_embedding, # at t-1
                        dynamic_item_embedding,# at t-1
                        features,
                        delta_u):
        concat_input = torch.concat([
                                    dynamic_item_embedding,
                                    features,
                                    delta_u.reshape(-1,1)]
                                    ,axis=1)
        
 

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
        concatenated_input = torch.concat((u_projection,u_static,i_dynamic,i_static),axis=1)
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
                next_item_dynamic_embedding,
                next_item_static_embedding
                ):
      ######## New  Dynamic Embeddings ########
      # New dynamic embedding of the user
      future_user_embedding= self.update_user_rnn(actual_user_embedding,actual_item_embedding,f,delta_u)
      # New dynamic embedding of the item
      future_item_embedding= self.update_item_rnn(actual_item_embedding,actual_user_embedding,f,delta_i)     

      projected_user_embedding = self.projection_operation(future_user_embedding,delta_i)


      if self.option =="interaction_prediction":

        j_tilde = self.predict_item_embedding(
          projected_user_embedding,
          u_static,
          future_item_embedding,
          i_static)
      
        # The real next item embedding j_true, is the concatenation of the static and dynamic embedding of the next item 
        j_true = torch.concat([next_item_dynamic_embedding,next_item_static_embedding],axis=1)

        # Return loss value between the predicted embedding "j_tilde" and the real next item embedding j_true
        loss = MSELoss()(j_tilde,j_true)
        loss += regularizer(actual_user_embedding,future_user_embedding,self.lambda_u,
                               actual_item_embedding,future_item_embedding,self.lambda_i
                               )
        


      else:
        # Prediction of next state of the user using an MLP at the end
        u_tilde = self.predict_user_state(projected_user_embedding)
        loss = CrossEntropyLoss()(u_tilde,next_state_label)

      return future_user_embedding,future_item_embedding,loss
