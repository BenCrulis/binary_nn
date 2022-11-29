from abc import ABC, abstractmethod


class Logger(ABC):
    @abstractmethod
    def log(self, iteration, epoch, data, **kwargs):
        pass


class ConsoleLogger(Logger):
    def __init__(self, verbosity=0):
        self.verbosity = verbosity

    def log(self, iteration, epoch, data=None, level=1):
        data = {} if data is None else data
        if level <= self.verbosity:
            print(f"epoch {epoch}, iteration {iteration}:")
            for k, v in data.items():
                print(f"{k}: {v}")


class LoggerGroup(Logger):
    def __init__(self, loggers=None):
        self.loggers = loggers if loggers is not None else []

    def add_logger(self, logger):
        self.loggers.append(logger)

    def log(self, iteration, epoch, data, **kwargs):
        for logger in self.loggers:
            logger.log(iteration, epoch, data, **kwargs)


class WandbLogger(Logger):

    def __init__(self, project_name=None, run_name=None):
        import wandb
        wandb.init(project=project_name, name=run_name)

    def log(self, iteration, epoch, data, **kwargs):
        import wandb
        logged = {"epoch": epoch, **data}
        wandb.log(logged, step=iteration)
