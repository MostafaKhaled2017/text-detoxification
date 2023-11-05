import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class ParaNMTVisualizer:
    def __init__(self, data_path, figures_path='figures'):
        self.data = pd.read_csv(data_path)
        self.figures_path = figures_path
        if not os.path.exists(self.figures_path):
            os.makedirs(self.figures_path)

    def save_figure(self, plt, fig_name):
        """Save the figure with a given name in the figures folder."""
        plt.savefig(f'reports/{self.figures_path}/{fig_name}.png')

    def plot_word_frequency(self, column, top_n=20):
        """Plot and save the frequency of the top N words in a given column."""
        all_words = ' '.join(self.data[column].astype(str)).split()
        freq_dist = pd.Series(all_words).value_counts()[:top_n]
        plt.figure(figsize=(12, 8))
        sns.barplot(x=freq_dist.values, y=freq_dist.index)
        plt.title(f'Top {top_n} most frequent words in {column}')
        plt.xlabel('Frequency')
        plt.ylabel('Words')
        self.save_figure(plt, f'word_frequency_{column}')

    def plot_sentence_length(self, column):
        """Plot and save the distribution of sentence lengths for a given column."""
        self.data[f'{column}_len'] = self.data[column].astype(str).apply(lambda x: len(x.split()))
        plt.figure(figsize=(12, 8))
        sns.histplot(self.data[f'{column}_len'], kde=True)
        plt.title(f'Sentence Length Distribution in {column}')
        plt.xlabel('Sentence Length')
        plt.ylabel('Frequency')
        self.save_figure(plt, f'sentence_length_{column}')

    def scatter_plot_comparison(self, column1, column2):
        """Scatter plot and save to compare sentence lengths between two columns."""
        self.data[f'{column1}_len'] = self.data[column1].astype(str).apply(lambda x: len(x.split()))
        self.data[f'{column2}_len'] = self.data[column2].astype(str).apply(lambda x: len(x.split()))
        plt.figure(figsize=(12, 8))
        sns.scatterplot(x=f'{column1}_len', y=f'{column2}_len', data=self.data)
        plt.title(f'Comparison of Sentence Lengths: {column1} vs {column2}')
        plt.xlabel(f'{column1} Length')
        plt.ylabel(f'{column2} Length')
        self.save_figure(plt, f'scatter_plot_{column1}_vs_{column2}')

# Usage
visualizer = ParaNMTVisualizer('data/interim/01_ParaNMT_cleaned.csv')
visualizer.plot_word_frequency(column='reference')
visualizer.plot_sentence_length(column='reference')
visualizer.scatter_plot_comparison(column1='reference', column2='translation')