FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PDG_FILE_LEVEL_OUTPUT_DIR=/app/output

ARG SCANSIBLE_REPOSITORY=https://github.com/softwarelanguageslab/scansible.git
ARG SCANSIBLE_REF=

WORKDIR /app

RUN apt-get -o Acquire::Retries=5 update \
    && apt-get -o Acquire::Retries=5 install -y --no-install-recommends \
        ca-certificates \
        git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --upgrade pip \
    && python -m pip install --default-timeout=300 --retries=10 --no-cache-dir -r requirements.txt

RUN git clone "${SCANSIBLE_REPOSITORY}" /scansible \
    && if [ -n "${SCANSIBLE_REF}" ]; then git -C /scansible checkout "${SCANSIBLE_REF}"; fi \
    && python -m pip install --default-timeout=300 --retries=10 --no-cache-dir --no-deps /scansible

RUN python -c "from pathlib import Path; from tempfile import TemporaryDirectory; from ansible.parsing.yaml.objects import AnsibleBaseYAMLObject; import scansible, networkx as nx, pydot; tmp=TemporaryDirectory(); p=Path(tmp.name)/'test.dot'; p.write_text('digraph { 1 [label=\"a\"]; 1 -> 2 [label=DEF]; }'); g=nx.nx_pydot.read_dot(str(p)); assert g.number_of_nodes() == 2 and g.number_of_edges() == 1"

COPY . .

ENTRYPOINT ["python"]
CMD ["scripts/pdg_file_level_extractor.py", "--help"]
