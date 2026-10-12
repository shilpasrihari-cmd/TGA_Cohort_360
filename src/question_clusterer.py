"""Complete-link cosine clusters; model vectors are inputs, no LLM generation."""
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics.pairwise import cosine_similarity

class QuestionClusterer:
    def __init__(self, similarity_threshold=.70, review_threshold=.80):
        if not 0 < similarity_threshold <= review_threshold <= 1:
            raise ValueError('Similarity thresholds must satisfy 0 < grouping <= review <= 1.')
        self.similarity_threshold = similarity_threshold
        self.review_threshold = review_threshold

    def cluster(self, questions, embeddings):
        if len(questions) != len(embeddings):
            raise ValueError('Question rows and embedding rows must match.')
        result = pd.DataFrame(index=questions.index, columns=['cluster_id', 'cluster_review', 'cluster_min_similarity'])
        if questions.empty:
            return result
        vectors = np.asarray(embeddings)
        if vectors.ndim != 2 or not np.isfinite(vectors).all() or (np.linalg.norm(vectors, axis=1) == 0).any():
            raise ValueError('Embeddings must be finite nonzero vectors.')
        if questions.message_id.duplicated().any():
            raise ValueError('Question IDs must be unique.')
        groups = []
        # Restrict to a classified topic to avoid silently merging topic boundaries.
        for topic in sorted(questions.topic.unique()):
            positions = np.flatnonzero(questions.topic.to_numpy() == topic)
            positions = sorted(positions, key=lambda pos: str(questions.iloc[pos].message_id))
            similarity = np.clip(cosine_similarity(vectors[positions]), -1, 1)
            distance = np.clip(1-similarity, 0, 2)
            np.fill_diagonal(distance, 0)
            labels = (fcluster(linkage(squareform(distance, checks=False), method='complete'),
                      t=1-self.similarity_threshold, criterion='distance')
                      if len(positions) > 1 else np.array([0]))
            for label in sorted(set(labels)):
                local = np.flatnonzero(labels == label)
                members = [positions[i] for i in local]
                minimum = float(similarity[np.ix_(local, local)].min())
                review = len(members) > 1 and minimum < self.review_threshold
                if len(members) == 1 and len(positions) > 1:
                    nearest = float(np.delete(similarity[local[0]], local[0]).max())
                    review = nearest >= self.similarity_threshold-.10
                groups.append((min(str(questions.iloc[p].message_id) for p in members), members, minimum, review))
        for number, (_, members, minimum, review) in enumerate(sorted(groups), start=1):
            for pos in members:
                result.iloc[pos] = [f'C{number:04}', bool(review), minimum]
        result['cluster_review'] = result.cluster_review.astype(bool)
        result['cluster_min_similarity'] = result.cluster_min_similarity.astype(float)
        return result
