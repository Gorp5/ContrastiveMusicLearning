# Contrastive Music Learning

This project develops Transformer-based contrastive learning models to cluster music. The goal is to learn meaningful representations of audio that can be used for various downstream tasks, such as music recommendation and genre classification.

## Architecture

The architecture is centered around a Transformer model that learns audio representations through contrastive learning. These representations are then used for clustering.

### Positional Embedding Variants

The Transformer model utilizes several positional embedding variants to encode the sequential nature of audio data:

*   **1D and 2D ALiBi (Attention with Linear Biases):** ALiBi biases attention scores based on the distance between tokens, providing a simple and effective way to incorporate positional information. This project uses both 1D and 2D versions of ALiBi.
*   **RoPE (Rotary Position Embedding):** RoPE encodes absolute positional information by rotating the query and key vectors. This method has been shown to be effective in various Transformer models.

### Clustering

The learned audio representations are clustered using the following methods:

*   **Hierarchical Clustering with Elki:** The project uses the [ELKI](https://elki-project.github.io/) data mining toolkit to perform hierarchical clustering. This is the primary clustering method used in the `ClusterTesting.ipynb` notebook.
*   **K-means, DBScan:** While the prompt mentioned k-means and DBScan, they are not currently implemented in the project.

### Spotify Integration

The project includes analysis of Spotify data in the `DataAnalysis.ipynb` notebook. However, it does not feature a direct wrapper for the Spotify API. Instead, it processes and analyzes a pre-existing dataset of song features from a CSV file.

## Playlist Generation

Playlist generation is a potential application of this project, but it is not yet implemented. The use of convex hulls for playlist generation, as mentioned in the prompt, is a possible future direction for this work.

## To-Do and Future Work

*   **K-means Loss:** The prompt mentioned a to-do with k-means loss, but this is not yet implemented in the codebase. A k-means-based loss function could be explored to improve the clustering performance of the learned representations.
*   **Implement Additional Clustering Algorithms:** K-means and DBScan could be implemented to provide alternative clustering methods.
*   **Spotify API Wrapper:** A direct integration with the Spotify API would allow for real-time data retrieval and playlist generation.
*   **Playlist Generation with Convex Hulls:** Implementing playlist generation, potentially using convex hulls to create diverse and coherent playlists, would be a valuable addition.
