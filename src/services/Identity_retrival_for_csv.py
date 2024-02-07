from langchain.document_loaders.csv_loader import CSVLoader
from app import utils
from app.llm import Llm
from components.base_component import BaseComponent
from datalayer.Neo4jDumper import Neo4jDumper
from langchain.document_loaders.csv_loader import CSVLoader
import pandas as pd

import logging
from langchain.document_loaders.csv_loader import CSVLoader
import pandas as pd

class SafeCSVLoader(CSVLoader):
    def __init__(self, file_path, logger=None):
        super().__init__(file_path=file_path)
        self.logger = logger if logger else logging.getLogger(__name__)
        # Set up basic logging if no logger is provided
        if logger is None:
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    def load(self):
        encodings_to_try = ['utf-8', 'ISO-8859-1', 'Windows-1252']
        for encoding in encodings_to_try:
            try:
                return pd.read_csv(self.file_path, encoding=encoding)
            except UnicodeDecodeError as e:
                self.logger.error(f"编码错误使用 {encoding}: {e}，文件：{self.file_path}")
        raise RuntimeError(f"无法使用以下编码读取文件 {self.file_path}: {encodings_to_try}")



class NameIdentityRetrievalForCsv(BaseComponent):
    def __init__(self, model_name, data_path):
        super().__init__('NameIdentityRetrievalForCsv')
        self.sources = utils.read_yaml_file(data_path)
        self.csv_sources = self.sources.get('csv', [])
        self.neo4j_instance = Neo4jDumper(config_path='config.yml')
        self.open_ai_llm = Llm(model=model_name)

    def run(self, **kwargs):
        for csvfile in self.csv_sources:
            try:
                loader = SafeCSVLoader(file_path=csvfile, logger=self.logger)
                data = loader.load()
                if data is None:
                    continue

                self.logger.info(f'Processing data from {csvfile}')
                # Process the data as needed...
                # ...

                # After processing, send data to OpenAI and Neo4j
                self.logger.info(f'Extracting knowledge graph using OpenAI model for data source: {csvfile}')
                response = self.open_ai_llm.run(input_text=data.iloc[-1])

                if response == -1:
                    self.logger.error(f"OpenAI LLM returned an error for {csvfile}")
                    continue
                self.neo4j_instance.run(data=response)
                self.logger.info(f'Knowledge graph populated successfully for data source: {csvfile}')

            except Exception as e:
                # This will log the full exception traceback, giving you more information about the error
                self.logger.exception(f"处理文件 {csvfile} 时发生错误")
                continue


# Don't forget to call your main function
if __name__ == "__main__":
    # Replace 'your_model_name' and 'your_data_path' with the actual values you want to use.
    component = NameIdentityRetrievalForCsv(model_name='your_model_name', data_path='your_data_path')
    component.run()

            # # setting up openai model and extracting knowledge graph
            # self.logger.info(f'loading model {self.open_ai_llm}')
            # # just sending last few lines of csv as the token limit is limited of openai api free version.
            # # model should  be changed to claude2 (Anthropic) or premium openai api key should be used.
            # # response = self.open_ai_llm.extract_and_store_graph(document=data[-1])
            # response = self.open_ai_llm.run(input_text=data[-1])
            # # instantiating neo4jBD and dumping the knowledge graph
            # self.neo4j_instance.run(data=response)
            # self.logger.info(f'knowledge graph populated successfully for data source: {csvfile}')
