FROM continuumio/anaconda3
RUN conda update -y python
RUN apt-get update && apt-get install -y python3-dev gcc
RUN pip install scikit-learn pandas plotly xgboost keras
EXPOSE 8888
WORKDIR /app
CMD ["jupyter", "notebook", "--allow-root", "--ip", "0.0.0.0", "--NotebookApp.token=''"]
