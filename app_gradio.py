import pandas as pd
import gradio as gr
import matplotlib.pyplot as plt
import seaborn as sns
import src.Connectors.gcp_bq_queries as gcp_bq_py
import src.ClassicML.DGE.pydeseq_utils as pydeseq_utils


def generate_histogram(project_id, dataset_id, table_id, primary_site, primary_diagnosis):
    bq_queries = gcp_bq_py.BigQueryQueries(project_id, dataset_id, table_id)
    value_counts = bq_queries.get_all_primary_diagnosis_for_primary_site(primary_site) 
    # Generate histogram data for primary diagnosis in tumor tissue type
    primary_diagnosis_histogram = value_counts.copy()    
    sns.set_theme(rc={'figure.figsize':(19.7,15.27)})
    ax = sns.barplot(data=primary_diagnosis_histogram, y='primary_diagnosis', x='number_of_cases', orient='h')
    # fig, ax = plt.subplots()
    # primary_diagnosis_histogram.plot(kind='bar', ax=ax)
    ax.set_xlabel('Primary Diagnosis')
    ax.set_ylabel('Count')
    ax.set_title('Histogram of Primary Diagnoses in Tumor Tissue Type')
    
    plt.xticks(rotation=90)
    fig = ax.figure.get_figure()
    
    return fig

def retrieve_data(project_id, dataset_id, table_id, primary_site, primary_diagnosis):
    bq_queries = gcp_bq_py.BigQueryQueries(project_id, dataset_id, table_id)
    df = bq_queries.query_data(primary_site, primary_diagnosis)
    return df 

def update_primary_diagnosis_options(project_id, dataset_id, table_id, primary_site):
    bq_queries = gcp_bq_py.BigQueryQueries(project_id, dataset_id, table_id)
    return bq_queries.get_primary_diagnosis_options(primary_site)

def pydeseq2_analysis(project_id, dataset_id, table_id, primary_site, primary_diagnosis):
    # Placeholder for PyDESeq2 analysis
    bq_queries = gcp_bq_py.BigQueryQueries(project_id, dataset_id, table_id)
    df = bq_queries.query_data(primary_site, primary_diagnosis)
        
    return "PyDESeq2 analysis result"

def supervised_ml_analysis(project_id, dataset_id, table_id, primary_site, primary_diagnosis):
    # Placeholder for Supervised ML analysis
    return "Supervised ML analysis result"

def unsupervised_learning_analysis(project_id, dataset_id, table_id, primary_site, primary_diagnosis):
    # Placeholder for Unsupervised Learning analysis
    return "Unsupervised Learning analysis result"

def gradio_interface():
    with gr.Blocks() as demo:
        with gr.Tab("Data Retrieval"):
            with gr.Row():
                project_id = gr.Textbox(label="Google Cloud Project ID")
                dataset_id = gr.Textbox(label="BigQuery Dataset ID")
                table_id = gr.Textbox(label="BigQuery Table ID")
                primary_site = gr.Textbox(label="Primary Site")
                primary_diagnosis = gr.Dropdown(label="Primary Diagnosis", choices=[])
            
            get_primary_diagnosis_button = gr.Button("Get Primary Diagnosis Options")
            
            def update_primary_diagnosis_options_wrapper(project_id, dataset_id, table_id, primary_site):
                return gr.update(choices=update_primary_diagnosis_options(project_id, dataset_id, table_id, primary_site))
            
            get_primary_diagnosis_button.click(
                update_primary_diagnosis_options_wrapper,
                inputs=[project_id, dataset_id, table_id, primary_site],
                outputs=primary_diagnosis
            )
            
            retrieve_button = gr.Button("Generate Histogram")
            primary_diagnosis_histogram = gr.Plot(label="Primary Diagnosis Histogram")
            retrieve_button.click(
                generate_histogram,
                inputs=[project_id, dataset_id, table_id, primary_site, primary_diagnosis],
                outputs=[primary_diagnosis_histogram]
            )
        
        with gr.Tab("Analysis"):
            with gr.Row():
                pydeseq_button = gr.Button("Run PyDESeq2 Analysis")
                supervised_ml_button = gr.Button("Run Supervised ML Analysis")
                unsupervised_ml_button = gr.Button("Run Unsupervised Learning Analysis")
                
            pydeseq_result = gr.Textbox(label="PyDESeq2 Result")
            supervised_ml_result = gr.Textbox(label="Supervised ML Result")
            unsupervised_ml_result = gr.Textbox(label="Unsupervised Learning Result")
            
            pydeseq_button.click(
                pydeseq2_analysis,
                inputs=[project_id, dataset_id, table_id, primary_site, primary_diagnosis],
                outputs=pydeseq_result
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

