# MLOps Monitoring

This directory contains tools and configurations for monitoring our ML systems.

## Directory Structure

- `evidently/` - Data drift monitoring with Evidently AI
- `prometheus/` - Metrics collection and storage
- `grafana/` - Dashboards and visualization

## Evidently AI Data Drift Monitoring

The `evidently/` directory contains:

- `drift_detection.py` - Script for detecting data/concept drift in ML models

To run drift detection:

```bash
python monitoring/evidently/drift_detection.py --drift-percent 0.2 --output-dir ./drift_reports
```

The script will:
1. Load reference data (training data)
2. Generate simulated production data with specified drift
3. Run drift detection analysis
4. Save HTML reports and JSON metrics

## Prometheus Monitoring

The `prometheus/` directory contains:

- Configuration for collecting metrics from:
  - Model API service (prediction counts, latency)
  - Kubernetes infrastructure
  - Other MLOps services

Prometheus is deployed via Kubernetes manifests in the `infrastructure/kubernetes/` directory.

Access the Prometheus UI:

```bash
minikube service prometheus -n mlops
```

## Grafana Dashboards

The `grafana/` directory contains:

- Dashboard definitions for:
  - Model performance monitoring
  - Data drift visualization
  - System metrics

Grafana is deployed via Kubernetes manifests in the `infrastructure/kubernetes/` directory.

Access the Grafana UI:

```bash
minikube service grafana -n mlops
```

Default credentials:
- Username: `admin`
- Password: `admin`

## Setting Up Alerts

To set up alerting:

1. Configure AlertManager in Prometheus
2. Create alert rules in Grafana
3. Configure notification channels (email, Slack, etc.)

Example alert scenarios:

- Data drift exceeds threshold
- Model prediction latency spikes
- Error rate increases
- Resource utilization (CPU/memory) exceeds limits 