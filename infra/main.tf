provider "google" {
  project = var.project_id
  region  = var.region
}

resource "google_cloud_run_service" "ml_api" {
  name     = "mlops-api"
  location = var.region

  template {
    spec {
      containers {
        image = var.image_url
      }
    }
  }
}

output "cloud_run_url" {
  value = google_cloud_run_service.ml_api.status[0].url
}
