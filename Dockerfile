FROM python:3.12

RUN apt-get update && apt-get install -y \
    git \
    graphviz

RUN pip install "ansible-core==2.15.12"

RUN pip install \
    click \
    attrs \
    attrs-strict \
    graphviz \
    jinja2 \
    kuzu \
    loguru \
    pydantic \
    requests \
    rich \
    rustworkx

RUN git clone https://github.com/softwarelanguageslab/scansible.git /scansible

WORKDIR /scansible

RUN pip install --no-deps .

RUN pip install \
    networkx \
    PyYAML \
    matplotlib \
    pandas \
    gitpython \
    pydot

WORKDIR /app