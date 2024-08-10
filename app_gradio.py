import pandas as pd
import gradio as gr
import matplotlib.pyplot as plt
import seaborn as sns
import src.Connectors.gcp_bq_queries as gcp_bq_py
import src.ClassicML.DGE.pydeseq_utils as pydeseq_utils
import src.Engines.analysis_engine as an_eng
import pandas as pd
from gseapy.plot import gseaplot
import gseapy as gp
from gseapy import dotplot
import matplotlib.pyplot as plt
import numpy as np


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

def retrieve_data(project_id, dataset_id, table_id, primary_site, primary_diagnosis):
    bq_queries = gcp_bq_py.BigQueryQueries(project_id, dataset_id, table_id)
    df = bq_queries.get_df_for_pydeseq(primary_site, primary_diagnosis)
    data_from_bq = df.copy()
    analysis_cls = an_eng.Analysis(data_from_bq, analysis_type='DE')
    gene_ids_or_gene_cols = list(pd.read_csv('/Users/abhilashdhal/Projects/personal_docs/Transcriptomics/data/gene_annotation/gene_id_to_gene_name_mapping.csv')['gene_id'])
    exp_data = analysis_cls.expand_data_from_bq(data_from_bq, gene_ids_or_gene_cols, 'DE')
    counts_for_de = analysis_cls.counts_from_bq_df(exp_data, gene_ids_or_gene_cols)
    metadata = analysis_cls.metadata_for_pydeseq(exp_data)
    return counts_for_de, metadata 

def update_primary_site_options(project_id, dataset_id, table_id):
    # Assume gcp_bq_py.BigQueryQueries has a method to get primary site options
    bq_queries = gcp_bq_py.BigQueryQueries(project_id, dataset_id, table_id)
    return bq_queries.get_primary_site_options()

def update_primary_diagnosis_options(project_id, dataset_id, table_id, primary_site):
    bq_queries = gcp_bq_py.BigQueryQueries(project_id, dataset_id, table_id)
    return bq_queries.get_primary_diagnosis_options(primary_site)

def pydeseq2_analysis(project_id, dataset_id, table_id, primary_site, primary_diagnosis):
    counts_for_de, metadata = retrieve_data(project_id, dataset_id, table_id, primary_site, primary_diagnosis)
    gene_ids_or_gene_cols = pd.read_csv('/Users/abhilashdhal/Projects/personal_docs/Transcriptomics/data/gene_annotation/gene_id_to_gene_name_mapping.csv')
    analysis_cls = an_eng.Analysis(data_from_bq=None, analysis_type='DE')
    res_pydeseq = analysis_cls.run_pydeseq(metadata=metadata, counts=counts_for_de)
    res_pydeseq_with_gene_names = pd.merge(res_pydeseq, gene_ids_or_gene_cols, left_on='index', right_on='gene_id')
    return res_pydeseq_with_gene_names 

def gsea_analysis(res_pydeseq, gene_set):
    analysis_cls = an_eng.Analysis(data_from_bq=None, analysis_type='DE')
    result, plot = analysis_cls.run_gsea(res_pydeseq, gene_set) 
    return result, plot

def update_gsea_options():
    gsea_options = gp.get_library_name()
    return gsea_options

def supervised_ml_analysis(project_id, dataset_id, table_id, primary_site, primary_diagnosis):
    # Placeholder for Supervised ML analysis
    return "Supervised ML analysis result"

def unsupervised_learning_analysis(project_id, dataset_id, table_id, primary_site, primary_diagnosis):
    # Placeholder for Unsupervised Learning analysis
    return "Unsupervised Learning analysis result"

def gradio_interface():
    with gr.Blocks() as demo:
        # data_storage = gr.Variable(value=pd.DataFrame())
        with gr.Row():
            logo_url = "OmixhubLogo.png"
            gr.HTML(f'<div style="text-align: center;"><img src="file/{logo_url}" alt="OmixHub Logo" style="width: 100px;"><h1>OmixHub</h1></div>')

        with gr.Tab("Cohort Selection"):
            with gr.Row():
                project_id = gr.Textbox(label="Google Cloud Project ID")
                dataset_id = gr.Textbox(label="BigQuery Dataset ID")
                table_id = gr.Textbox(label="BigQuery Table ID")
                primary_site = gr.Dropdown(label="Primary Site", choices=[])
                primary_diagnosis = gr.Dropdown(label="Primary Diagnosis", choices=[])
            
            
            get_primary_site_button = gr.Button("Get Primary Site Options")
            
            def update_primary_site_options_wrapper(project_id, dataset_id, table_id):
                options = update_primary_site_options(project_id, dataset_id, table_id)
                return gr.update(choices=options)
            
            get_primary_site_button.click(
                update_primary_site_options_wrapper,
                inputs=[project_id, dataset_id, table_id],
                outputs=[primary_site]
            )
            get_primary_diagnosis_button = gr.Button("Get Primary Diagnosis Options")
            def update_primary_diagnosis_options_wrapper(project_id, dataset_id, table_id, primary_site):
                options = update_primary_diagnosis_options(project_id, dataset_id, table_id, primary_site)
                return gr.update(choices=options)
            
            get_primary_diagnosis_button.click(
                update_primary_diagnosis_options_wrapper,
                inputs=[project_id, dataset_id, table_id, primary_site],
                outputs=[primary_diagnosis]
            )

 
            retrieve_button2 = gr.Button("Generate Histogram")
            primary_diagnosis_histogram = gr.Plot(label="Primary Diagnosis Histogram")
            retrieve_button2.click(
                generate_histogram,
                inputs=[project_id, dataset_id, table_id, primary_site, primary_diagnosis],
                outputs=[primary_diagnosis_histogram]
            )

        
        with gr.Tab("Analysis"):
            with gr.Row():
                pydeseq_button = gr.Button("Run PyDESeq2 Analysis")
                supervised_ml_button = gr.Button("Run Supervised ML Analysis")
                unsupervised_ml_button = gr.Button("Run Unsupervised Learning Analysis")
                gene_set = gr.Dropdown(label="Gene Sets", choices=[])
            
            ### Run PyDeSeq    
            pydeseq_results_df = gr.Dataframe(label="PyDESeq2 Results", interactive=True)
            download_link = gr.File(label="Download Results CSV", visible=False)

            supervised_ml_result = gr.Textbox(label="Supervised ML Result")
            unsupervised_ml_result = gr.Textbox(label="Unsupervised Learning Result")
            
            pydeseq_button.click(
                pydeseq2_analysis,
                inputs=[project_id, dataset_id, table_id, primary_site, primary_diagnosis],
                outputs=pydeseq_results_df
            )

                        
            get_gsea_gene_set_button = gr.Button("Select Gene Set")
            def update_gsea_options_wrapper():
                options = update_gsea_options()
                return gr.update(choices=options)
            
            get_gsea_gene_set_button.click(
                update_gsea_options_wrapper,
                inputs=None,
                outputs=[gene_set]
            )
            
            gsea_plot = gr.Plot(label="GSEA Enrichment Plot") 
            gsea_result = gr.Dataframe(label='Gene set Enrichments')
            run_gsea_button = gr.Button("Run Gene Set Enrichment Analysis")
            run_gsea_button.click(
                gsea_analysis,
                inputs=[pydeseq_results_df, gene_set],
                outputs=[gsea_result, gsea_plot]
            )                       
            supervised_ml_button.click(
                supervised_ml_analysis,
                inputs=[project_id, dataset_id, table_id, primary_site, primary_diagnosis],
                outputs=supervised_ml_result
            )
            unsupervised_ml_button.click(
                unsupervised_learning_analysis,
                inputs=[project_id, dataset_id, table_id, primary_site, primary_diagnosis],
                outputs=unsupervised_ml_result
            )

    demo.launch()

if __name__ == "__main__":
    gradio_interface()