const SHARED_ABNORMALITIES = [
  "Glioma Probability",
  "Meningioma Signal",
  "Pituitary Lesion",
  "Edema Response",
  "Hemorrhage Marker",
  "Necrosis Pattern",
  "Mass Effect",
  "Calcification Risk",
  "Malignancy Index"
];

const MODEL_PAGE_CONFIG = {
  "brain-mri": {
    key: "brain-mri",
    title: "AI-Powered Brain MRI Analysis",
    subtitle: "VGG-16 multi-head inference for neuro-oncology screening and localization.",
    icon: "🧠",
    serviceLabel: "Brain MRI",
    themeAccent: "var(--accent-primary)",
    status: "Ready",
    modelInfo: {
      Architecture: "VGG-16 + 9 Heads",
      Dataset: "BraTS + Internal Curated MRI",
      "Input Size": "224 x 224",
      Version: "v2.8.3",
      Throughput: "2.4 scans/sec"
    },
    abnormalitySchema: SHARED_ABNORMALITIES,
    layoutVariant: "list",
    heatmapSlots: 9,
    summaryDefaults: {
      meanScore: 0,
      elevatedFeatures: 0,
      regionsFound: 0,
      modelStatus: "Ready"
    }
  },
  "ct-scan": {
    key: "ct-scan",
    title: "AI-Enhanced CT Scan Analysis",
    subtitle: "Cross-sectional anomaly scoring and explainable heatmap generation.",
    icon: "🔬",
    serviceLabel: "CT Scan",
    themeAccent: "var(--accent-secondary)",
    status: "Ready",
    modelInfo: {
      Architecture: "VGG-16 + Dense Fusion",
      Dataset: "NIH CT + Internal Multi-Organ",
      "Input Size": "256 x 256",
      Version: "v3.1.0",
      Throughput: "2.1 scans/sec"
    },
    abnormalitySchema: [
      "Pulmonary Opacity",
      "Nodule Suspicion",
      "Pleural Effusion",
      "Consolidation Pattern",
      "Fibrosis Signature",
      "Emphysema Index",
      "Severity Composite"
    ],
    layoutVariant: "bento",
    heatmapSlots: 8,
    summaryDefaults: {
      meanScore: 0,
      elevatedFeatures: 0,
      regionsFound: 0,
      modelStatus: "Ready"
    }
  },
  xray: {
    key: "xray",
    title: "Smart X-Ray Interpretation Suite",
    subtitle: "Rapid triage signals with visual attribution and confidence overlays.",
    icon: "🩻",
    serviceLabel: "X-Ray",
    themeAccent: "var(--accent-tertiary)",
    status: "Ready",
    modelInfo: {
      Architecture: "VGG-16 + Attention Head",
      Dataset: "CheXpert + Internal X-Ray",
      "Input Size": "224 x 224",
      Version: "v2.5.4",
      Throughput: "3.2 scans/sec"
    },
    abnormalitySchema: [
      "Pneumonia Risk",
      "Cardiomegaly Index",
      "Pleural Thickening",
      "Atelectasis Signal",
      "Fracture Marker",
      "Infiltration Pattern",
      "Effusion Indicator",
      "Opacity Density",
      "Acute Alert Score",
      "Pleural Effusion Marker",
      "Interstitial Pattern",
      "Pulmonary Edema Signal"
    ],
    layoutVariant: "list",
    heatmapSlots: 11,
    summaryDefaults: {
      meanScore: 0,
      elevatedFeatures: 0,
      regionsFound: 0,
      modelStatus: "Ready"
    }
  }
};

window.MODEL_PAGE_CONFIG = MODEL_PAGE_CONFIG;
