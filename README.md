# overview
  This program is a general-purpose ranking system for document ranking, which includes natural language processing and deep interest network models. Recently, it has achieved an MRR (Mean Reciprocal Rank) value of 0.82, demonstrating its effectiveness.

# features
  + Implemented using the TF-Ranking framework published by Google, which provides a high-level API for building ranking models.
  + Utilizes list-wise loss to optimize the ranking model, which is a more effective approach than point-wise loss.
  + Scores a document based on the user's recent reading history, paying more attention to similar documents. This approach improves the relevance of the search results and enhances the user experience.