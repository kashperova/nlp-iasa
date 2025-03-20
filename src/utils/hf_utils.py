import evaluate
import numpy as np
import wandb
from transformers import TrainerCallback


seqeval_metric = evaluate.load("seqeval")


def compute_ner_metrics(out: tuple, id2label: dict[int, str]) -> dict[str, float]:
    predictions, labels = out
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval_metric.compute(
        predictions=true_predictions, references=true_labels
    )
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


class SaveArtifactCallback(TrainerCallback):
    def __init__(self, model_name: str, save_interval: int):
        self.model_name = model_name
        self.save_interval = save_interval

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step > 0 and state.global_step % self.save_interval == 0:
            artifact = wandb.Artifact(
                f"{self.model_name}-{state.global_step}", type="model"
            )
            artifact.add_dir(args.output_dir)
            wandb.log_artifact(artifact)
            print(f"Logged artifact at step {state.global_step}")

        return control
