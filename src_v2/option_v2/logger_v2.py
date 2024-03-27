from timeit import default_timer as timer


def log_text(file_path, log):
    if not log.endswith('\n'):
        log += '\n'

    print(log)
    with open(file_path, 'a') as f:
        f.write(log)


def log_args(file_path, args):
    log = f"Args: {args}\n"
    log_text(file_path, log)


def log_train_epoch(file_path, epoch, train_loss):
    log = f"Epoch: {epoch}, Train loss: {train_loss:.3f}\n"
    log_text(file_path, log)


def log_val_epoch(file_path, epoch, val_loss):
    log = f"Epoch: {epoch}, Val loss: {val_loss:.3f}\n"
    log_text(file_path, log)


def log_test_metrics(file_path, precision, recall, f1, accuracy, cm):
    log = (f"Precision: {precision}\n"
           f"Recall: {recall}\n"
           f"F1 score: {f1}\n"
           f"Accuracy: {accuracy}\n"
           f"Confusion Matrix:\n{cm}\n")
    log_text(file_path, log)


def log_target_test_metrics(file_path, target, loss_fn):
    log = (f"{target}:\n"
           f"\tPrecision: {round(loss_fn, 4)}\n")
    log_text(file_path, log)
