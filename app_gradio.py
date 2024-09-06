from src.Engines import analysis_engine as an_eng
# Connectors.gcp_bigquery_utils, ClassicML.DGE.pydeseq_utils
import src.Connectors.gcp_bigquery_utils as gcp_bq_py
import src.ClassicML.DGE.pydeseq_utils as pydeseq_utils
from src.ClassicML.DataAug import simulators

import pandas as pd
import gradio as gr
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from gseapy.plot import gseaplot
import gseapy as gp
from gseapy import dotplot
import matplotlib.pyplot as plt
import numpy as np
import tempfile
import os
# Assuming gene set options are static or fetched from a file
# GENE_SETS = pd.read_csv('/path_to_gene_sets/gene_sets.csv')['gene_set'].tolist()

def simulate_normal_samples(project_id, dataset_id, table_id, primary_site, primary_diagnosis, num_samples):
    bq_queries = gcp_bq_py.BigQueryQueries(project_id, dataset_id, table_id)
    df = bq_queries.get_df_for_pydeseq(primary_site, primary_diagnosis)
    
    simulator = simulators.AutoencoderSimulator(df)
    preprocessed_data = simulator.preprocess_data()
    simulator.train_autoencoder(preprocessed_data)
    simulated_samples = simulator.simulate_samples(num_samples)

    # Convert simulated_samples to a dataframe if it's not already
    if not isinstance(simulated_samples, pd.DataFrame):
        simulated_samples = pd.DataFrame(simulated_samples)

    return simulated_samples, simulated_samples

def generate_histogram(project_id, dataset_id, table_id, primary_site, primary_diagnosis):
    bq_queries = gcp_bq_py.BigQueryQueries(project_id, dataset_id, table_id)
    value_counts = bq_queries.get_all_primary_diagnosis_for_primary_site(primary_site) 
    primary_diagnosis_histogram = value_counts.copy()    
    sns.set_theme(rc={'figure.figsize':(19.7,15.27)})
    ax = sns.barplot(data=primary_diagnosis_histogram, y='primary_diagnosis', x='number_of_cases', orient='h')
    ax.set_xlabel('Primary Diagnosis')
    ax.set_ylabel('Count')
    ax.set_title('Histogram of Primary Diagnoses in Tumor Tissue Type')
    
    plt.xticks(rotation=90)
    fig = ax.figure.get_figure()
    
    return fig

def generate_tissue_type_barplot(project_id, dataset_id, table_id, primary_site, primary_diagnosis):
    bq_queries = gcp_bq_py.BigQueryQueries(project_id, dataset_id, table_id)
    df = bq_queries.get_df_for_pydeseq(primary_site, primary_diagnosis)
    
    tissue_type_counts = df['tissue_type'].value_counts()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=tissue_type_counts.index, y=tissue_type_counts.values, ax=ax)
    
    ax.set_xlabel('Tissue Type')
    ax.set_ylabel('Count')
    ax.set_title(f'Count of Tumor and Normal Samples for {primary_diagnosis} in {primary_site}')
    
    for i, v in enumerate(tissue_type_counts.values):
        ax.text(i, v, str(v), ha='center', va='bottom')
    
    plt.tight_layout()
    return fig

def retrieve_data(project_id, dataset_id, table_id, primary_site, primary_diagnosis, simulated_samples=None):
    bq_queries = gcp_bq_py.BigQueryQueries(project_id, dataset_id, table_id)
    df = bq_queries.get_df_for_pydeseq(primary_site, primary_diagnosis)
    data_from_bq = df.copy()
    
    if simulated_samples is not None:
        # Assuming simulated_samples has the same structure as data_from_bq
        data_from_bq = pd.concat([data_from_bq, simulated_samples], ignore_index=True)
    
    analysis_cls = an_eng.AnalysisEngine(data_from_bq, analysis_type='DE')
    if not analysis_cls.check_tumor_normal_counts():
        raise ValueError("Tumor and Normal counts should be at least 10 each")
    gene_ids_or_gene_cols = list(pd.read_csv('/Users/abhilashdhal/Projects/personal_docs/data/Transcriptomics/data/gene_annotation/gene_id_to_gene_name_mapping.csv')['gene_id'])
    exp_data = analysis_cls.expand_data_from_bq(data_from_bq, gene_ids_or_gene_cols, 'DE')
    counts_for_de = analysis_cls.counts_from_bq_df(exp_data, gene_ids_or_gene_cols)
    metadata = analysis_cls.metadata_for_pydeseq(exp_data)
    return counts_for_de, metadata 

def pydeseq2_analysis(project_id, dataset_id, table_id, primary_site, primary_diagnosis, simulated_samples):
    counts_for_de, metadata = retrieve_data(project_id, dataset_id, table_id, primary_site, primary_diagnosis, simulated_samples)
    gene_ids_or_gene_cols = pd.read_csv('/Users/abhilashdhal/Projects/personal_docs/data/Transcriptomics/data/gene_annotation/gene_id_to_gene_name_mapping.csv')
    analysis_cls = an_eng.AnalysisEngine(data_from_bq=None, analysis_type='DE')
    res_pydeseq = analysis_cls.run_pydeseq(metadata=metadata, counts=counts_for_de)
    res_pydeseq_with_gene_names = pd.merge(res_pydeseq, gene_ids_or_gene_cols, left_on='index', right_on='gene_id')
    return res_pydeseq_with_gene_names 

def update_primary_site_options(project_id, dataset_id, table_id):
    # Assume gcp_bq_py.BigQueryQueries has a method to get primary site options
    bq_queries = gcp_bq_py.BigQueryQueries(project_id, dataset_id, table_id)
    return bq_queries.get_primary_site_options()

def update_primary_diagnosis_options(project_id, dataset_id, table_id, primary_site):
    bq_queries = gcp_bq_py.BigQueryQueries(project_id, dataset_id, table_id)
    return bq_queries.get_primary_diagnosis_options(primary_site)

def gsea_analysis(res_pydeseq_with_gene_names, gene_set):
    analysis_cls = an_eng.AnalysisEngine(data_from_bq=None, analysis_type='DE')
    out_df, axs, pre_res = analysis_cls.run_gsea(res_pydeseq_with_gene_names, gene_set)
    
    # Generate GSEA plot
    if isinstance(axs, plt.Figure):
        gsea_fig = axs
    elif isinstance(axs, list) and len(axs) > 0:
        gsea_fig = axs[0].figure
    else:
        gsea_fig = plt.figure()
        plt.text(0.5, 0.5, "GSEA plot not available", ha='center', va='center')

    gsea_fig.suptitle("GSEA Enrichment Plot")
    plt.tight_layout()
    
    # Generate Enrichment Map
    em_fig, _ = analysis_cls.plot_enrichment_map(pre_res)
    em_fig.suptitle("Enrichment Map")
    plt.tight_layout()

    # Generate Dotplot
    dot_fig, _ = analysis_cls.create_dotplot(pre_res, cutoff=1.0, figsize=(10, 12))
    # dot_fig.suptitle("GSEA Dotplot")
    # plt.tight_layout()

    # Save figures to files
    gsea_fig.savefig('gsea_plot.png')
    em_fig.savefig('enrichment_map.png')
    dot_fig.savefig('dotplot.png')

    # Close figures to free up memory
    plt.close(gsea_fig)
    plt.close(em_fig)
    plt.close(dot_fig)

    return out_df, 'gsea_plot.png', 'enrichment_map.png', 'dotplot.png'

def update_gsea_options():
    gsea_options = gp.get_library_name()
    return gsea_options

def supervised_ml_analysis(project_id, dataset_id, table_id, primary_site, primary_diagnosis):
    # Placeholder for Supervised ML analysis
    return "Supervised ML analysis result"

def unsupervised_learning_analysis(project_id, dataset_id, table_id, primary_site, primary_diagnosis):
    # Placeholder for Unsupervised Learning analysis
    return "Unsupervised Learning analysis result"

def save_plot(plot_data):
    if plot_data is not None:
        # Create a new figure and axis
        fig, ax = plt.subplots()
        
        # Check if plot_data has 'data' attribute (for scatter plots)
        if hasattr(plot_data, 'data'):
            for trace in plot_data.data:
                ax.plot(trace.x, trace.y, label=trace.name)
            ax.legend()
        # Check if plot_data has 'layout' attribute (for other plot types)
        elif hasattr(plot_data, 'layout'):
            # You might need to adjust this part based on the specific plot type
            ax.imshow(plot_data.layout.data[0].z)
        
        # Set title if available
        if hasattr(plot_data, 'layout') and hasattr(plot_data.layout, 'title'):
            ax.set_title(plot_data.layout.title.text)
        
        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        
        # Save the figure to the temporary file
        plt.savefig(temp_file.name)
        plt.close(fig)  # Close the figure to free up memory
        
        return temp_file.name
    return None

def save_dataframe(df):
    if df is not None and not df.empty:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
        df.to_csv(temp_file.name, index=False)
        return temp_file.name
    return None

def update_primary_diagnosis_options_wrapper(project_id, dataset_id, table_id, primary_site):
    options = update_primary_diagnosis_options(project_id, dataset_id, table_id, primary_site)
    return gr.update(choices=options)

def update_primary_site_options_wrapper(project_id, dataset_id, table_id):
    options = update_primary_site_options(project_id, dataset_id, table_id)
    return gr.update(choices=options)

def plot_similarity_heatmap(project_id, dataset_id, table_id, primary_site, primary_diagnosis):
    bq_queries = gcp_bq_py.BigQueryQueries(project_id, dataset_id, table_id)
    df = bq_queries.get_df_for_pydeseq(primary_site, primary_diagnosis)
    
    simulator =simulators.AutoencoderSimulator(df)
    fig = simulator.plot_similarity_heatmap()
    return fig

def gradio_interface():
    with gr.Blocks() as demo:
        # data_storage = gr.Variable(value=pd.DataFrame())
        with gr.Row():
            logo_url = "OmixhubLogo.png"
            gr.HTML(f'<div style="text-align: center;"><img src="file/{logo_url}" alt="OmixHub Logo" style="width: 100px;"><h1>OmixHub</h1></div>')

        simulated_samples = gr.State(None)
        with gr.Tab("Cohort Selection"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Cohort Retrieval from GDC")
                    project_id = gr.Textbox(label="Google Cloud Project ID")
                    dataset_id = gr.Textbox(label="BigQuery Dataset ID")
                    table_id = gr.Textbox(label="BigQuery Table ID")
                    primary_site = gr.Dropdown(label="Primary Site", choices=[])
                    primary_diagnosis = gr.Dropdown(label="Primary Diagnosis", choices=[])
                    
                    get_primary_site_button = gr.Button("Get Primary Site Options")
                    get_primary_diagnosis_button = gr.Button("Get Primary Diagnosis Options")
                    
                    retrieve_button1 = gr.Button("Generate Primary Diagnosis Histogram")
                    primary_diagnosis_histogram = gr.Plot(label="Primary Diagnosis Histogram")
                    
                    retrieve_button2 = gr.Button("Generate Tissue Type Barplot")
                    tissue_type_barplot = gr.Plot(label="Tissue Type Barplot")

                with gr.Column(scale=1):
                    gr.Markdown("### Simulate Normal Tissue Samples")
                    num_samples = gr.Slider(minimum=1, maximum=1000, step=1, label="Number of Samples to Simulate", value=100)
                    simulate_button = gr.Button("Simulate Normal Samples")
                    simulated_samples_df = gr.Dataframe(label="Simulated Normal Samples")
                    download_simulated_samples = gr.Button("Download Simulated Samples")

                    gr.Markdown("### Similarity Heatmap")
                    generate_heatmap_button = gr.Button("Generate Similarity Heatmap")
                    similarity_heatmap = gr.Plot(label="Similarity Heatmap")

            # Event handlers for the left column (Cohort Retrieval)
            get_primary_site_button.click(
                update_primary_site_options_wrapper,
                inputs=[project_id, dataset_id, table_id],
                outputs=[primary_site]
            )
            
            get_primary_diagnosis_button.click(
                update_primary_diagnosis_options_wrapper,
                inputs=[project_id, dataset_id, table_id, primary_site],
                outputs=[primary_diagnosis]
            )

            retrieve_button1.click(
                generate_histogram,
                inputs=[project_id, dataset_id, table_id, primary_site, primary_diagnosis],
                outputs=[primary_diagnosis_histogram]
            )

            retrieve_button2.click(
                generate_tissue_type_barplot,
                inputs=[project_id, dataset_id, table_id, primary_site, primary_diagnosis],
                outputs=[tissue_type_barplot]
            )
            
            # Event handlers for the right column (Simulate Normal Samples)
            def update_simulated_samples(samples):
                return samples
            # Event handlers for the right column (Simulate Normal Samples)
            simulate_button.click(
                simulate_normal_samples,
                inputs=[project_id, dataset_id, table_id, primary_site, primary_diagnosis, num_samples],
                outputs=[simulated_samples_df, simulated_samples]
            )

            download_simulated_samples.click(
                save_dataframe,
                inputs=[simulated_samples_df],
                outputs=[gr.File(label="Download Simulated Samples")]
            )

            generate_heatmap_button.click(
                plot_similarity_heatmap,
                inputs=[project_id, dataset_id, table_id, primary_site, primary_diagnosis],
                outputs=[similarity_heatmap]
            )
        
        with gr.Tab("Analysis"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Bioinformatics Analysis")
                    
                    # PyDESeq2 Section
                    pydeseq_button = gr.Button("Run PyDESeq2 Analysis")
                    pydeseq_results_df = gr.Dataframe(label="PyDESeq2 Results", interactive=True)
                    pydeseq_download = gr.Button("Download PyDESeq2 Results")
                    
                    # GSEA Section
                    gene_set = gr.Dropdown(label="Gene Sets", choices=update_gsea_options(), multiselect=True)
                    gsea_button = gr.Button("Run Gene Set Enrichment Analysis")
                    gsea_result = gr.Dataframe(label='Gene Set Enrichments')
                    gsea_plot = gr.Image(label="GSEA Enrichment Plot")
                    em_plot = gr.Image(label="Enrichment Map")
                    dot_plot = gr.Image(label="Dotplot")
                    gsea_plot_download = gr.Button("Download GSEA Plot")
                    gsea_result_download = gr.Button("Download GSEA Results")
                
                with gr.Column(scale=1):
                    gr.Markdown("### Unsupervised Learning")
                    
                    unsupervised_ml_button = gr.Button("Run Unsupervised Learning Analysis")
                    unsupervised_ml_result = gr.Textbox(label="Unsupervised Learning Result")
                    # Add more components for unsupervised learning as needed
                
                with gr.Column(scale=1):
                    gr.Markdown("### Supervised Learning")
                    
                    supervised_ml_button = gr.Button("Run Supervised ML Analysis")
                    supervised_ml_result = gr.Textbox(label="Supervised ML Result")
                    # Add more components for supervised learning as needed

            # PyDESeq2 functionality
            pydeseq_button.click(
                pydeseq2_analysis,
                inputs=[project_id, dataset_id, table_id, primary_site, primary_diagnosis, simulated_samples],
                outputs=pydeseq_results_df
            )
            pydeseq_download.click(
                save_dataframe,
                inputs=[pydeseq_results_df],
                outputs=[gr.File(label="Download PyDESeq2 Results")]
            )

            # GSEA functionality
            gsea_button.click(
                gsea_analysis,
                inputs=[pydeseq_results_df, gene_set],
                outputs=[gsea_result, gsea_plot, em_plot, dot_plot]
            )

            # gsea_button.click(gsea_analysis, 
            #                 inputs=[pydeseq_results_df, gene_set], 
            #                 outputs=[gsea_result, 
            #                         gr.Image(label="GSEA Enrichment Plot"), 
            #                         gr.Image(label="Enrichment Map"), 
            #                         gr.Image(label="GSEA Dotplot")])
            gsea_plot_download.click(
                lambda x: x,
                inputs=[gsea_plot],
                outputs=[gr.File(label="Download GSEA Plot")]
            )
            gsea_result_download.click(
                save_dataframe,
                inputs=[gsea_result],
                outputs=[gr.File(label="Download GSEA Results")]
            )

            # Unsupervised Learning functionality
            unsupervised_ml_button.click(
                unsupervised_learning_analysis,
                inputs=[project_id, dataset_id, table_id, primary_site, primary_diagnosis],
                outputs=unsupervised_ml_result)
           
            # Supervised Learning functionality
            supervised_ml_button.click(
                supervised_ml_analysis,
                inputs=[project_id, dataset_id, table_id, primary_site, primary_diagnosis],
                outputs=supervised_ml_result
            )

    demo.launch()

if __name__ == "__main__":
    gradio_interface()