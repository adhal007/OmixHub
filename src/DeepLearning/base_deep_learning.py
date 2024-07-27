class BaseDL:
    def __init__(self, model_name, model_dict) -> None:
        dft_model_dict = {
            'model_name': None,
            'model_location': None,
            'output_df': None
        }

    def get_importances(self):
        raise NotImplementedError()
    
    def get_output_clf_results(self):
        raise NotImplementedError()
    
    def cons_results(self):
        raise NotImplementedError()