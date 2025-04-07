import io
import base64
import logging
from typing import Dict, List, Any, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Set the style for all visualizations
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

def generate_visualization(df: pd.DataFrame, viz_type: str, columns: List[str]=None, parameters: Dict[str, Any]=None) -> str:
    """
    Generate a visualization for a dataset
    
    Args:
        df: DataFrame to visualize
        viz_type: Type of visualization to create
        columns: List of columns to include in the visualization
        parameters: Additional parameters for the visualization
        
    Returns:
        Base64 encoded image data
    """
    if parameters is None:
        parameters = {}
    
    if columns is None or len(columns) == 0:
        # Use numeric columns by default
        columns = df.select_dtypes(include=np.number).columns.tolist()[:5]  # Limit to 5 columns
    
    # Create a figure
    plt.figure(figsize=parameters.get("figsize", (10, 6)))
    
    try:
        # Generate the appropriate visualization
        if viz_type == "histogram":
            _generate_histogram(df, columns, parameters)
        elif viz_type == "boxplot":
            _generate_boxplot(df, columns, parameters)
        elif viz_type == "scatter":
            _generate_scatter(df, columns, parameters)
        elif viz_type == "correlation":
            _generate_correlation_heatmap(df, columns, parameters)
        elif viz_type == "pairplot":
            _generate_pairplot(df, columns, parameters)
        elif viz_type == "line":
            _generate_line_plot(df, columns, parameters)
        elif viz_type == "bar":
            _generate_bar_chart(df, columns, parameters)
        elif viz_type == "pie":
            _generate_pie_chart(df, columns, parameters)
        elif viz_type == "pca":
            _generate_pca_plot(df, columns, parameters)
        elif viz_type == "count":
            _generate_count_plot(df, columns, parameters)
        elif viz_type == "distribution":
            _generate_distribution_plot(df, columns, parameters)
        else:
            raise ValueError(f"Unsupported visualization type: {viz_type}")
        
        # Convert plot to base64 encoded string
        img_data = _fig_to_base64()
        return img_data
        
    except Exception as e:
        logger.error(f"Error generating {viz_type} visualization: {str(e)}")
        
        # Generate an error message figure instead
        plt.figure(figsize=(8, 4))
        plt.text(0.5, 0.5, f"Error generating visualization:\n{str(e)}",
                horizontalalignment='center', verticalalignment='center', fontsize=12)
        plt.axis('off')
        
        img_data = _fig_to_base64()
        return img_data

def _fig_to_base64() -> str:
    """Convert current figure to base64 encoded string"""
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()
    return img_str

def _generate_histogram(df: pd.DataFrame, columns: List[str], parameters: Dict[str, Any]):
    """Generate histograms for selected columns"""
    bins = parameters.get("bins", 30)
    kde = parameters.get("kde", True)
    
    if len(columns) == 1:
        # Single column histogram
        sns.histplot(df[columns[0]], bins=bins, kde=kde)
        plt.title(f"Histogram of {columns[0]}")
        plt.xlabel(columns[0])
        plt.ylabel("Frequency")
    else:
        # Multiple column histograms
        num_cols = min(len(columns), 4)  # Max 4 columns
        fig, axes = plt.subplots(num_cols, 1, figsize=(10, 3*num_cols))
        
        if num_cols == 1:
            axes = [axes]
            
        for i, col in enumerate(columns[:num_cols]):
            sns.histplot(df[col], bins=bins, kde=kde, ax=axes[i])
            axes[i].set_title(f"Histogram of {col}")
            axes[i].set_xlabel(col)
            axes[i].set_ylabel("Frequency")
        
        plt.tight_layout()

def _generate_boxplot(df: pd.DataFrame, columns: List[str], parameters: Dict[str, Any]):
    """Generate box plots for selected columns"""
    if "group_by" in parameters:
        # Grouped boxplot
        group_col = parameters["group_by"]
        sns.boxplot(x=group_col, y=columns[0], data=df)
        plt.title(f"Boxplot of {columns[0]} by {group_col}")
    else:
        # Regular boxplots
        sns.boxplot(data=df[columns])
        plt.title("Boxplot of Selected Features")
        plt.ylabel("Value")
        plt.xticks(rotation=45)

def _generate_scatter(df: pd.DataFrame, columns: List[str], parameters: Dict[str, Any]):
    """Generate scatter plot for two columns"""
    if len(columns) < 2:
        raise ValueError("Scatter plot requires at least 2 columns")
    
    x_col = columns[0]
    y_col = columns[1]
    
    if "hue" in parameters and parameters["hue"] in df.columns:
        hue_col = parameters["hue"]
        sns.scatterplot(x=x_col, y=y_col, hue=hue_col, data=df)
        plt.title(f"Scatter Plot: {x_col} vs {y_col} (colored by {hue_col})")
    else:
        sns.scatterplot(x=x_col, y=y_col, data=df)
        plt.title(f"Scatter Plot: {x_col} vs {y_col}")
    
    plt.xlabel(x_col)
    plt.ylabel(y_col)

def _generate_correlation_heatmap(df: pd.DataFrame, columns: List[str], parameters: Dict[str, Any]):
    """Generate correlation heatmap for selected columns"""
    corr_matrix = df[columns].corr()
    
    # Generate mask for the upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    
    # Draw the heatmap
    sns.heatmap(
        corr_matrix, 
        mask=mask, 
        cmap=cmap, 
        vmax=1, 
        vmin=-1, 
        center=0,
        annot=True, 
        fmt=".2f",
        square=True, 
        linewidths=.5, 
        cbar_kws={"shrink": .5}
    )
    
    plt.title("Correlation Matrix")
    plt.tight_layout()

def _generate_pairplot(df: pd.DataFrame, columns: List[str], parameters: Dict[str, Any]):
    """Generate pair plot for selected columns"""
    # Limit to 5 columns max for readability
    vis_columns = columns[:5]
    
    if "hue" in parameters and parameters["hue"] in df.columns:
        hue_col = parameters["hue"]
        g = sns.pairplot(df[vis_columns + [hue_col]], hue=hue_col)
    else:
        g = sns.pairplot(df[vis_columns])
    
    g.fig.suptitle("Pair Plot Matrix", y=1.02)
    plt.tight_layout()

def _generate_line_plot(df: pd.DataFrame, columns: List[str], parameters: Dict[str, Any]):
    """Generate line plot (assumes first column is x-axis)"""
    if len(columns) < 2:
        raise ValueError("Line plot requires at least 2 columns")
        
    x_col = parameters.get("x", columns[0])
    
    # Sort by x column if it's a time or numeric column
    if pd.api.types.is_numeric_dtype(df[x_col]):
        df_sorted = df.sort_values(by=x_col)
    else:
        df_sorted = df
        
    plt.figure(figsize=(10, 6))
    
    # Plot each y column
    for col in columns:
        if col != x_col:
            plt.plot(df_sorted[x_col], df_sorted[col], marker='o', linestyle='-', label=col)
    
    plt.title(f"Line Plot by {x_col}")
    plt.xlabel(x_col)
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    
    # Rotate x labels if there are many
    if df[x_col].nunique() > 10:
        plt.xticks(rotation=45)
    
    plt.tight_layout()

def _generate_bar_chart(df: pd.DataFrame, columns: List[str], parameters: Dict[str, Any]):
    """Generate bar chart for selected columns"""
    if "group_by" in parameters and parameters["group_by"] in df.columns:
        # Grouped bar chart
        group_col = parameters["group_by"]
        
        if len(columns) > 1:
            # Multiple columns to plot
            melted_df = df.melt(
                id_vars=[group_col], 
                value_vars=[col for col in columns if col != group_col],
                var_name='Variable', 
                value_name='Value'
            )
            sns.barplot(x=group_col, y="Value", hue="Variable", data=melted_df)
            plt.title(f"Bar Chart by {group_col}")
            plt.xticks(rotation=45)
        else:
            # Single column with grouping
            agg_df = df.groupby(group_col)[columns[0]].mean().reset_index()
            sns.barplot(x=group_col, y=columns[0], data=agg_df)
            plt.title(f"Bar Chart of {columns[0]} by {group_col}")
            plt.xticks(rotation=45)
    else:
        # Simple bar chart of column values
        plt.figure(figsize=(12, 6))
        
        for i, col in enumerate(columns):
            if df[col].nunique() <= 20:  # Only for categorical or low-cardinality
                value_counts = df[col].value_counts().sort_values(ascending=False)
                top_n = min(10, len(value_counts))  # Limit to top 10
                
                plt.subplot(len(columns), 1, i+1)
                sns.barplot(x=value_counts.index[:top_n], y=value_counts.values[:top_n])
                plt.title(f"Top {top_n} values for {col}")
                plt.xticks(rotation=45)
                
        plt.tight_layout()

def _generate_pie_chart(df: pd.DataFrame, columns: List[str], parameters: Dict[str, Any]):
    """Generate pie chart for a categorical column"""
    if len(columns) == 0:
        raise ValueError("Pie chart requires at least one column")
    
    col = columns[0]
    value_counts = df[col].value_counts()
    
    # Limit number of slices for readability
    if len(value_counts) > 8:
        # Keep top 7 and group others
        top_n = value_counts.iloc[:7]
        others = pd.Series({'Others': value_counts.iloc[7:].sum()})
        value_counts = pd.concat([top_n, others])
    
    plt.pie(
        value_counts.values, 
        labels=value_counts.index, 
        autopct='%1.1f%%', 
        startangle=90
    )
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    plt.title(f"Distribution of {col}")

def _generate_pca_plot(df: pd.DataFrame, columns: List[str], parameters: Dict[str, Any]):
    """Generate PCA plot for dimensionality reduction visualization"""
    # PCA needs numeric data
    numeric_cols = df[columns].select_dtypes(include=np.number).columns
    
    if len(numeric_cols) < 2:
        raise ValueError("PCA plot requires at least 2 numeric columns")
    
    # Scale the data
    X = StandardScaler().fit_transform(df[numeric_cols])
    
    # Apply PCA to reduce to 2 dimensions
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X)
    
    # Create a DataFrame with the principal components
    pca_df = pd.DataFrame(
        data=principal_components, 
        columns=['PC1', 'PC2']
    )
    
    # Add a color variable if specified
    if "hue" in parameters and parameters["hue"] in df.columns:
        pca_df[parameters["hue"]] = df[parameters["hue"]].values
        sns.scatterplot(x='PC1', y='PC2', hue=parameters["hue"], data=pca_df)
        plt.title(f"PCA Plot (colored by {parameters['hue']})")
    else:
        sns.scatterplot(x='PC1', y='PC2', data=pca_df)
        plt.title("PCA Plot")
    
    # Add explained variance ratio
    explained_variance = pca.explained_variance_ratio_
    plt.xlabel(f"PC1 ({explained_variance[0]:.2%} variance)")
    plt.ylabel(f"PC2 ({explained_variance[1]:.2%} variance)")

def _generate_count_plot(df: pd.DataFrame, columns: List[str], parameters: Dict[str, Any]):
    """Generate count plot for categorical variables"""
    if len(columns) == 0:
        raise ValueError("Count plot requires at least one column")
    
    if len(columns) == 1:
        # Single column
        sns.countplot(y=columns[0], data=df, order=df[columns[0]].value_counts().index)
        plt.title(f"Count of {columns[0]}")
        plt.tight_layout()
    else:
        # Multiple columns
        fig, axes = plt.subplots(len(columns), 1, figsize=(10, 4*len(columns)))
        
        for i, col in enumerate(columns):
            if df[col].nunique() <= 30:  # Only for low-cardinality
                if len(columns) == 1:
                    ax = axes
                else:
                    ax = axes[i]
                    
                sns.countplot(y=col, data=df, order=df[col].value_counts().index, ax=ax)
                ax.set_title(f"Count of {col}")
                
        plt.tight_layout()

def _generate_distribution_plot(df: pd.DataFrame, columns: List[str], parameters: Dict[str, Any]):
    """Generate distribution plot for numeric variables"""
    if len(columns) == 0:
        raise ValueError("Distribution plot requires at least one column")
    
    # Make sure we have numeric columns
    numeric_cols = [col for col in columns if pd.api.types.is_numeric_dtype(df[col])]
    
    if not numeric_cols:
        raise ValueError("Distribution plot requires numeric columns")
    
    if len(numeric_cols) == 1:
        # Single distribution
        sns.kdeplot(df[numeric_cols[0]], shade=True)
        plt.title(f"Distribution of {numeric_cols[0]}")
    else:
        # Multiple distributions
        for col in numeric_cols:
            sns.kdeplot(df[col], label=col)
            
        plt.title("Distribution of Features")
        plt.legend()
    
    plt.xlabel("Value")
    plt.ylabel("Density") 