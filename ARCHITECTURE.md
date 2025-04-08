# System Architecture

```mermaid
graph TB
    subgraph "Data Layer"
        D1[Raw Data]
        D2[Feature Store]
        D3[Processed Data]
        D4[MinIO Storage]
    end

    subgraph "ML Pipeline"
        P1[Data Validation]
        P2[Feature Engineering]
        P3[Model Training]
        P4[Model Evaluation]
        P5[MLflow Tracking]
    end

    subgraph "Deployment Layer"
        K1[Minikube Cluster]
        K2[Model API Service]
        K3[Training API Service]
    end

    subgraph "Monitoring & Observability"
        M1[Prometheus]
        M2[Grafana]
        M3[Loki]
        M4[EvidentlyAI]
    end

    subgraph "CI/CD Pipeline"
        C1[GitHub Actions]
        C2[Automated Testing]
        C3[Model Deployment]
    end

    subgraph "Metadata & Lineage"
        L1[OpenLineage]
        L2[DataHub]
    end

    %% Data Flow
    D1 --> P1
    P1 --> D2
    D2 --> P2
    P2 --> D3
    D3 --> P3
    P3 --> P4
    P4 --> P5
    P5 --> D4

    %% Deployment Flow
    D4 --> K1
    K1 --> K2
    K1 --> K3

    %% Monitoring Flow
    K2 --> M1
    K2 --> M3
    K2 --> M4
    M1 --> M2
    M3 --> M2

    %% CI/CD Flow
    C1 --> C2
    C2 --> C3
    C3 --> K1

    %% Lineage Flow
    P1 --> L1
    P2 --> L1
    P3 --> L1
    P4 --> L1
    L1 --> L2

    %% Styling
    classDef data fill:#f9f,stroke:#333,stroke-width:2px
    classDef pipeline fill:#bbf,stroke:#333,stroke-width:2px
    classDef deployment fill:#bfb,stroke:#333,stroke-width:2px
    classDef monitoring fill:#fbb,stroke:#333,stroke-width:2px
    classDef cicd fill:#ff9,stroke:#333,stroke-width:2px
    classDef lineage fill:#bff,stroke:#333,stroke-width:2px

    class D1,D2,D3,D4 data
    class P1,P2,P3,P4,P5 pipeline
    class K1,K2,K3 deployment
    class M1,M2,M3,M4 monitoring
    class C1,C2,C3 cicd
    class L1,L2 lineage
```

## Architecture Components

### Data Layer
- **Raw Data**: Initial input datasets
- **Feature Store**: Centralized feature storage and transformation
- **Processed Data**: Cleaned and transformed data ready for training
- **MinIO Storage**: S3-compatible storage for artifacts

### ML Pipeline
- **Data Validation**: Schema and statistical validation
- **Feature Engineering**: Consistent feature transformation
- **Model Training**: ML model development
- **Model Evaluation**: Performance assessment
- **MLflow Tracking**: Experiment tracking and model versioning

### Deployment Layer
- **Minikube Cluster**: Local Kubernetes environment
- **Model API Service**: FastAPI-based model serving
- **Training API Service**: API for triggering retraining

### Monitoring & Observability
- **Prometheus**: Metrics collection
- **Grafana**: Visualization and dashboards
- **Loki**: Log aggregation
- **EvidentlyAI**: Data drift detection

### CI/CD Pipeline
- **GitHub Actions**: Automated workflows
- **Automated Testing**: Unit, integration, and E2E tests
- **Model Deployment**: Automated deployment process

### Metadata & Lineage
- **OpenLineage**: Data and model lineage tracking
- **DataHub**: Metadata management and visualization

## Data Flow
1. Raw data undergoes validation and processing
2. Features are engineered and stored in the feature store
3. Models are trained and evaluated using MLflow
4. Artifacts are stored in MinIO
5. Models are deployed to the Minikube cluster
6. Performance is monitored through Prometheus and Grafana
7. Logs are collected by Loki
8. Data drift is monitored by EvidentlyAI
9. Lineage is tracked through OpenLineage and DataHub 