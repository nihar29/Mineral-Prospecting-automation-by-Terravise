FROM mambaorg/micromamba:1.5-jammy

WORKDIR /app
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml
RUN micromamba install -y -n base -f /tmp/environment.yml && micromamba clean -a -y

COPY --chown=$MAMBA_USER:$MAMBA_USER . /app
ARG MAMBA_DOCKERFILE_ACTIVATE=1

# Default: run the whole pipeline. Mount data/ and outputs/ as volumes.
ENTRYPOINT ["/usr/local/bin/_entrypoint.sh"]
CMD ["snakemake", "--cores", "4"]
