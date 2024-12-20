import torch
from model.general.attention.disentangled_multihead_self import DisentangledMultiHeadSelfAttention
from model.general.attention.additive import AdditiveAttention


class UserEncoder(torch.nn.Module):
    def __init__(self, config, **kwargs):
        super(UserEncoder, self).__init__()
        self.config = config
        #disentangled
        self.disentangled_multihead_self_attention = DisentangledMultiHeadSelfAttention(
            config.word_embedding_dim, config.num_attention_heads)
        #disentangled
        self.additive_attention = AdditiveAttention(config.query_vector_dim,
                                                    config.word_embedding_dim)\

        #disentangled
        self.n_items = kwargs.get('n_items')
        self.device = kwargs.get('device')
        self.news_embedding_layer = torch.nn.Embedding(
            self.n_items + 1,
            config.word_embedding_dim,
            padding_idx  = 0,
        )
        #disentangled

    #disentangled
    def forward(self, user_news_id, user_vector):
        """
        Args:
            user_vector: batch_size, num_clicked_news_a_user, word_embedding_dim
        Returns:
            (shape) batch_size, word_embedding_dim
            
        """
        # print(user_news_id.shape)
        # print(user_vector.shape)
        user_news_id = user_news_id.to(self.device)
        user_emb = self.news_embedding_layer(user_news_id)
        # batch_size, num_clicked_news_a_user, word_embedding_dim
        disentangled_multihead_user_vector = self.disentangled_multihead_self_attention(
            user_vector,
            user_emb,
        )
        # batch_size, word_embedding_dim
        final_user_vector = self.additive_attention(disentangled_multihead_user_vector)
        return final_user_vector
    #disentangled