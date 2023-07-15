import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.tree import _tree, DecisionTreeClassifier
from IPython.display import display, HTML

def plot_cluster_pca(data, indices, labels, column_names, subset_indices=None):
    # Crear un DataFrame a partir de los datos estandarizados
    df = pd.DataFrame(data, columns=column_names)
    
    # Realizar PCA para obtener los dos componentes principales más importantes
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(df)
    
    # Crear un nuevo DataFrame con los componentes principales y las etiquetas de los clusters
    pca_df = pd.DataFrame(data=principal_components, columns=['Componente 1', 'Componente 2'])
    pca_df['Cluster'] = labels
    
    # Obtener los nombres de los índices para mostrar en el gráfico
    index_names = indices
    
    # Crear una figura y un eje para el gráfico
    fig, ax = plt.subplots(figsize =(10, 7))

    # Iterar sobre cada etiqueta de cluster y asignar un color para el gráfico
    unique_labels = np.unique(labels)
    colors = plt.cm.get_cmap('tab10', len(unique_labels))
    for i, label in enumerate(unique_labels):
        cluster_indices = np.where(labels == label)[0]
        if subset_indices:
            cluster_indices = cluster_indices[subset_indices]
        ax.scatter(pca_df.loc[cluster_indices, 'Componente 1'], pca_df.loc[cluster_indices, 'Componente 2'],
                   c=[colors(i)], label=f'Cluster {label}')
    
    # Etiquetar los puntos del gráfico con los nombres de los índices
    #for i, name in enumerate(index_names):
        #ax.annotate(name, (pca_df.loc[indices[i], 'Componente 1'], pca_df.loc[indices[i], 'Componente 2']))
    
    # Mostrar leyenda y etiquetas de los ejes
    ax.legend()
    ax.set_xlabel(f'Componente 1 ({pca.explained_variance_ratio_[0]*100:.2f}% var. explicada)')
    ax.set_ylabel(f'Componente 2 ({pca.explained_variance_ratio_[1]*100:.2f}% var. explicada)')
    
    # Mostrar el gráfico
    plt.show()
    
    # Obtener la importancia de las variables en los componentes principales
    component_importance = pd.DataFrame(pca.components_.T, columns=['Componente 1', 'Componente 2'], index=column_names)
    
    # Imprimir el resumen de la importancia de las variables
    return component_importance


def pretty_print(df):
    return display( HTML( df.to_html().replace("\\n","<br>") ) )

def get_class_rules(tree: DecisionTreeClassifier, feature_names: list):
    inner_tree: _tree.Tree = tree.tree_
    classes = tree.classes_
    class_rules_dict = dict()

    def tree_dfs(node_id=0, current_rule=[]):
        # feature[i] holds the feature to split on, for the internal node i.
        split_feature = inner_tree.feature[node_id]
        if split_feature != _tree.TREE_UNDEFINED: # internal node
            name = feature_names[split_feature]
            threshold = inner_tree.threshold[node_id]
            # left child
            left_rule = current_rule + ["({} <= {})".format(name, threshold)]
            tree_dfs(inner_tree.children_left[node_id], left_rule)
            # right child
            right_rule = current_rule + ["({} > {})".format(name, threshold)]
            tree_dfs(inner_tree.children_right[node_id], right_rule)
        else: # leaf
            dist = inner_tree.value[node_id][0]
            dist = dist/dist.sum()
            max_idx = dist.argmax()
            if len(current_rule) == 0:
                rule_string = "ALL"
            else:
                rule_string = " and ".join(current_rule)
          # register new rule to dictionary
            selected_class = classes[max_idx]
            class_probability = dist[max_idx]
            class_rules = class_rules_dict.get(selected_class, [])
            class_rules.append((rule_string, class_probability))
            class_rules_dict[selected_class] = class_rules

    tree_dfs() # start from root, node_id = 0
    return class_rules_dict

def cluster_report(data: pd.DataFrame, clusters, min_samples_leaf=50, pruning_level=0.01):
    # Create Model
    tree = DecisionTreeClassifier(min_samples_leaf=min_samples_leaf, ccp_alpha=pruning_level)
    tree.fit(data, clusters)
    
    # Generate Report
    feature_names = data.columns
    class_rule_dict = get_class_rules(tree, feature_names)

    report_class_list = []
    for class_name in class_rule_dict.keys():
        rule_list = class_rule_dict[class_name]
        combined_string = ""
        for rule in rule_list:
            combined_string += "[{}] {}\n\n".format(rule[1], rule[0])
        report_class_list.append((class_name, combined_string))
        
    cluster_instance_df = pd.Series(clusters).value_counts().reset_index()
    cluster_instance_df.columns = ['class_name', 'instance_count']
    report_df = pd.DataFrame(report_class_list, columns=['class_name', 'rule_list'])
    report_df = pd.merge(cluster_instance_df, report_df, on='class_name', how='left')
    pretty_print(report_df.sort_values(by='class_name')[['class_name', 'instance_count', 'rule_list']])