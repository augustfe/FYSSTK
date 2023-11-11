import numpy as np
from Schedules import (
    Scheduler,
    Constant,
    Momentum,
    Adagrad,
    AdagradMomentum,
    Adam,
    RMS_prop,
    TimeDecay,
)
from Gradients import Gradients
from plotutils import (
    plotThetas,
    PlotPredictionPerVariable,
    PlotErrorPerVariable,
    plotHeatmap,
)
from typing import Callable
import pandas as pd
from sklearn.metrics import mean_squared_error


class GradAnalysis:
    def __init__(
        self,
        x_vals: np.ndarray,
        y_vals: np.ndarray,
        n_epochs: int = 500,
        seed: int = None,
        model: str = "OLS",
        method: str = "analytic",
        base_theta: np.ndarray = None,
        target_func: Callable = None,
        true_theta: np.ndarray = None,
        base_x: np.ndarray = None,
    ) -> None:
        self.x_vals = x_vals
        self.y_vals = y_vals
        self.n_epochs = n_epochs
        self.n_points = len(x_vals)
        self.seed = seed
        self.model = model
        self.method = method

        if self.seed is not None:
            np.random.seed(self.seed)

        if base_theta is None:
            self.base_theta = np.random.randn(3, 1)
        else:
            self.base_theta = base_theta.reshape(-1, 1)

        self.target_func = target_func
        self.true_theta = true_theta

        self.base_x = base_x
        if self.base_x is None:
            self.base_x = np.linspace(-2, 2, 100)

    def error_and_theta_vals_gd(
        self, schedulers: list[Scheduler], dim: int = 2
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        theta_arr = np.zeros((len(schedulers), dim + 1))
        error_arr = np.zeros((len(schedulers), self.n_epochs))

        for i, schedule in enumerate(schedulers):
            Gradient = Gradients(
                self.n_points,
                self.x_vals,
                self.y_vals,
                self.model,
                self.method,
                schedule,
            )

            theta_arr[i, :] = Gradient.GradientDescent(
                self.base_theta, self.n_epochs
            ).ravel()
            error_arr[i, :] = Gradient.errors

        return theta_arr, error_arr

    def error_per_variables(
        self, schedules: list[list[Scheduler]], dim: int = 2
    ) -> np.ndarray:
        error_arr = np.zeros((len(schedules), len(schedules[0])))
        ynew = self.target_func(self.base_x)

        for i, row in enumerate(schedules):
            for j, schedule in enumerate(row):
                Gradient = Gradients(
                    self.n_points,
                    self.x_vals,
                    self.y_vals,
                    self.model,
                    self.method,
                    schedule,
                )
                theta = Gradient.GradientDescent(self.base_theta, self.n_epochs)
                ypred = Gradient.predict(self.base_x, theta, dim)
                error_arr[i, j] = mean_squared_error(ynew, ypred)

        return error_arr

    def pred_per_theta(
        self, x: np.ndarray, theta_arr: np.ndarray, dim: int = 2
    ) -> np.ndarray:
        pred_arr = np.zeros((len(theta_arr), self.n_points))

        DummyGrad = Gradients(
            self.n_points,
            self.x_vals,
            self.y_vals,
            self.model,
            self.method,
            Constant(1),
        )
        for i, theta in enumerate(theta_arr):
            pred_arr[i, :] = DummyGrad.predict(x, theta, dim)

        return pred_arr

    def error_per_minibatch(
        self,
        schedule: Scheduler,
        minibatches: np.ndarray,
        n_epochs: int = 150,
        dim: int = 2,
    ) -> tuple:
        error_arr = np.zeros((len(minibatches), n_epochs))
        theta_arr = np.zeros((len(minibatches), dim + 1))
        for i, batch_size in enumerate(minibatches):
            Gradient = Gradients(
                self.n_points,
                self.x_vals,
                self.y_vals,
                self.model,
                self.method,
                schedule,
            )
            theta_arr[i, :] = Gradient.StochasticGradientDescent(
                self.base_theta, n_epochs, batch_size
            ).ravel()
            error_arr[i, :] = Gradient.errors

        return error_arr, theta_arr

    def constant_analysis(self, eta_vals: np.ndarray, dim: int = 2) -> None:
        schedulers = [Constant(eta) for eta in eta_vals]
        theta_arr, error_arr = self.error_and_theta_vals_gd(schedulers, dim)
        pred_arr = self.pred_per_theta(self.base_x, theta_arr, dim)

        PlotErrorPerVariable(
            error_arr,
            eta_vals,
            title="Error per epoch (Constant)",
        )
        PlotPredictionPerVariable(
            self.base_x,
            pred_arr,
            eta_vals,
            title="Predicted polynomials (Constant)",
            n_epochs=self.n_epochs,
            target_func=self.target_func,
        )
        plotThetas(
            theta_arr,
            eta_vals,
            title="Model parameters (Constant)",
            true_theta=self.true_theta,
        )

    def momentum_analysis(
        self,
        eta_vals: np.ndarray,
        rho_vals: np.ndarray,
        dim: int = 2,
        showPlots: bool = True,
        savePlots: bool = False,
    ) -> None:
        schedulers = [Momentum(eta, 0.9) for eta in eta_vals]
        theta_arr, error_arr = self.error_and_theta_vals_gd(schedulers, dim)
        pred_arr = self.pred_per_theta(self.base_x, theta_arr, dim)

        PlotErrorPerVariable(
            error_arr, eta_vals, title=r"Error per epoch (Momentum) $\rho=0.9$"
        )
        PlotPredictionPerVariable(
            self.base_x,
            pred_arr,
            eta_vals,
            title=r"Predicted polynomials (Momentum) $\rho=0.9$",
            n_epochs=self.n_epochs,
            target_func=self.target_func,
        )
        plotThetas(
            theta_arr,
            eta_vals,
            title=r"Model parameters (Momentum) $\rho=0.9$",
            true_theta=self.true_theta,
        )

        schedulers = [Momentum(0.001, rho) for rho in rho_vals]
        theta_arr, error_arr = self.error_and_theta_vals_gd(schedulers, dim)
        pred_arr = self.pred_per_theta(self.base_x, theta_arr, dim)

        PlotErrorPerVariable(
            error_arr,
            rho_vals,
            title=r"Error per epoch (Momentum) $\eta=0.001$",
            variable_label=r"$\rho$",
        )
        PlotPredictionPerVariable(
            self.base_x,
            pred_arr,
            rho_vals,
            title=r"Predicted polynomials (Momentum) $\eta=0.001$",
            n_epochs=self.n_epochs,
            target_func=self.target_func,
            variable_label=r"$\rho$",
        )
        plotThetas(
            theta_arr,
            rho_vals,
            title=r"Model parameters (Momentum) $\eta=0.001$",
            true_theta=self.true_theta,
            variable_label=r"$\rho$",
            variable_type="linear",
        )

        n_rho = 75
        n_eta = 75
        heat_rho = np.arctan(np.linspace(0, 10, n_rho))
        heat_rho = heat_rho / np.max(heat_rho)
        heat_eta = np.logspace(-7, -1, n_eta)

        schedulers = [[Momentum(eta, rho) for eta in heat_eta] for rho in heat_rho]
        error_arr = self.error_per_variables(schedulers, dim)

        df = pd.DataFrame(
            error_arr,
            index=[f"{rho:.2f}" for rho in heat_rho],
            columns=[f"{eta:.2e}" for eta in heat_eta],
        )
        plotHeatmap(
            df,
            title=f"Error after {self.n_epochs} epochs",
            x_label=r"$\eta$",
            y_label=r"$\rho$",
        )

    def adagrad_analysis(
        self,
        eta_vals: np.ndarray,
        dim: int = 2,
    ) -> None:
        schedulers = [Adagrad(eta) for eta in eta_vals]
        theta_arr, error_arr = self.error_and_theta_vals_gd(schedulers, dim)
        pred_arr = self.pred_per_theta(self.base_x, theta_arr, dim)

        PlotErrorPerVariable(
            error_arr,
            eta_vals,
            title=r"Error per epoch (Adagrad)",
        )
        PlotPredictionPerVariable(
            self.base_x,
            pred_arr,
            eta_vals,
            title=r"Predicted polynomials (Adagrad)",
            n_epochs=self.n_epochs,
            target_func=self.target_func,
        )
        plotThetas(
            theta_arr,
            eta_vals,
            title=r"Model parameters (Adagrad)",
            true_theta=self.true_theta,
        )

    def adagrad_momentum_analysis(
        self,
        eta_vals: np.ndarray,
        rho_vals: np.ndarray,
        dim: int = 2,
        showPlots: bool = True,
        savePlots: bool = False,
    ) -> None:
        schedulers = [AdagradMomentum(eta, 0.9) for eta in eta_vals]
        theta_arr, error_arr = self.error_and_theta_vals_gd(schedulers, dim)
        pred_arr = self.pred_per_theta(self.base_x, theta_arr, dim)

        PlotErrorPerVariable(
            error_arr,
            eta_vals,
            title=r"Error per epoch (AdagradMomentum) $\rho=0.9$",
        )
        PlotPredictionPerVariable(
            self.base_x,
            pred_arr,
            eta_vals,
            title=r"Predicted polynomials (AdagradMomentum) $\rho=0.9$",
            n_epochs=self.n_epochs,
            target_func=self.target_func,
        )
        plotThetas(
            theta_arr,
            eta_vals,
            title=r"Model parameters (AdagradMomentum) $\rho=0.9$",
            true_theta=self.true_theta,
        )

        schedulers = [AdagradMomentum(0.1, rho) for rho in rho_vals]
        theta_arr, error_arr = self.error_and_theta_vals_gd(schedulers, dim)
        pred_arr = self.pred_per_theta(self.base_x, theta_arr, dim)

        PlotErrorPerVariable(
            error_arr,
            rho_vals,
            title=r"Error per epoch (AdagradMomentum) $\eta=0.1$",
            variable_label=r"$\rho$",
        )
        PlotPredictionPerVariable(
            self.base_x,
            pred_arr,
            rho_vals,
            title=r"Predicted polynomials (AdagradMomentum) $\eta=0.1$",
            n_epochs=self.n_epochs,
            target_func=self.target_func,
            variable_label=r"$\rho$",
        )
        plotThetas(
            theta_arr,
            rho_vals,
            title=r"Model parameters (AdagradMomentum) $\eta=0.1$",
            true_theta=self.true_theta,
            variable_label=r"$\rho$",
            variable_type="linear",
        )

        n_rho = 75
        n_eta = 75
        heat_rho = np.arctan(np.linspace(0, 10, n_rho))
        # Scale rho to be between 0 and 1 (Want to remove the outliers just below 1)
        heat_rho = heat_rho * 2 / 3
        heat_eta = np.logspace(-3, 0, n_eta)

        schedulers = [
            [AdagradMomentum(eta, rho) for eta in heat_eta] for rho in heat_rho
        ]
        error_arr = self.error_per_variables(schedulers, dim)

        df = pd.DataFrame(
            error_arr,
            index=[f"{rho:.2f}" for rho in heat_rho],
            columns=[f"{eta:.2e}" for eta in heat_eta],
        )
        plotHeatmap(df, title=f"Error after {self.n_epochs} epochs")

    def adam_analysis(
        self,
        eta_vals: np.ndarray,
        rho_vals: np.ndarray,
        rho2_vals: np.ndarray,
        dim: int = 2,
        showPlots: bool = True,
        savePlots: bool = False,
    ) -> None:
        schedulers = [Adam(eta, 0.9, 0.999) for eta in eta_vals]
        theta_arr, error_arr = self.error_and_theta_vals_gd(schedulers, dim)
        pred_arr = self.pred_per_theta(self.base_x, theta_arr, dim)

        PlotErrorPerVariable(
            error_arr,
            eta_vals,
            title=r"Error per epoch (Adam) $\rho=0.9$, $\rho_2=0.999$",
        )
        PlotPredictionPerVariable(
            self.base_x,
            pred_arr,
            eta_vals,
            title=r"Predicted polynomials (Adam) $\rho=0.9$, $\rho_2=0.999$",
            n_epochs=self.n_epochs,
            target_func=self.target_func,
        )
        plotThetas(
            theta_arr,
            eta_vals,
            title=r"Model parameters (Adam) $\rho=0.9$, $\rho_2=0.999$",
            true_theta=self.true_theta,
        )

        schedulers = [Adam(0.1, rho, 0.999) for rho in rho_vals]
        theta_arr, error_arr = self.error_and_theta_vals_gd(schedulers, dim)
        pred_arr = self.pred_per_theta(self.base_x, theta_arr, dim)

        PlotErrorPerVariable(
            error_arr,
            rho_vals,
            title=r"Error per epoch (Adam) $\eta=0.1$, $\rho_2=0.999$",
            variable_label=r"$\rho$",
        )
        PlotPredictionPerVariable(
            self.base_x,
            pred_arr,
            rho_vals,
            title=r"Predicted polynomials (Adam) $\eta=0.1$, $\rho_2=0.999$",
            n_epochs=self.n_epochs,
            target_func=self.target_func,
            variable_label=r"$\rho$",
        )
        plotThetas(
            theta_arr,
            rho_vals,
            title=r"Model parameters (Adam) $\eta=0.1$, $\rho_2=0.999$",
            true_theta=self.true_theta,
            variable_label=r"$\rho$",
            variable_type="linear",
        )

        schedulers = [Adam(0.1, 0.9, rho2) for rho2 in rho2_vals]
        theta_arr, error_arr = self.error_and_theta_vals_gd(schedulers, dim)

        PlotErrorPerVariable(
            error_arr,
            rho2_vals,
            title=r"Error per epoch (Adam) $\eta=0.1$, $\rho=0.9$",
            variable_label=r"$\rho_2$",
        )
        plotThetas(
            theta_arr,
            rho2_vals,
            title=r"Model parameters (Adam) $\eta=0.1$, $\rho=0.9$",
            true_theta=self.true_theta,
            variable_label=r"$\rho_2$",
            variable_type="linear",
        )

        n_rho = 75
        n_eta = 75
        heat_rho = np.arctan(np.linspace(0.1, 10, n_rho))
        heat_rho = heat_rho * 2 / 3
        heat_eta = np.logspace(-3, 0, n_eta)

        schedulers = [[Adam(eta, rho, 0.999) for eta in heat_eta] for rho in heat_rho]
        error_arr = self.error_per_variables(schedulers, dim)

        df = pd.DataFrame(
            error_arr,
            index=[f"{rho:.2f}" for rho in heat_rho],
            columns=[f"{eta:.2e}" for eta in heat_eta],
        )
        plotHeatmap(df, title=f"Error after {self.n_epochs} epochs")

    def rms_prop_analysis(
        self,
        eta_vals: np.ndarray,
        rho_vals: np.ndarray,
        dim: int = 2,
        showPlots: bool = True,
        savePlots: bool = False,
    ) -> None:
        schedulers = [RMS_prop(eta, 0.9) for eta in eta_vals]
        theta_arr, error_arr = self.error_and_theta_vals_gd(schedulers, dim)
        pred_arr = self.pred_per_theta(self.base_x, theta_arr, dim)

        PlotErrorPerVariable(
            error_arr,
            eta_vals,
            title=r"Error per epoch (RMS_prop) $\rho=0.9$",
        )
        PlotPredictionPerVariable(
            self.base_x,
            pred_arr,
            eta_vals,
            title=r"Predicted polynomials (RMS_prop) $\rho=0.9$",
            n_epochs=self.n_epochs,
            target_func=self.target_func,
        )
        plotThetas(
            theta_arr,
            eta_vals,
            title=r"Model parameters (RMS_prop) $\rho=0.9$",
            true_theta=self.true_theta,
        )

        schedulers = [RMS_prop(0.01, rho) for rho in rho_vals]
        theta_arr, error_arr = self.error_and_theta_vals_gd(schedulers, dim)
        pred_arr = self.pred_per_theta(self.base_x, theta_arr, dim)

        PlotErrorPerVariable(
            error_arr,
            rho_vals,
            title=r"Error per epoch (RMS_prop) $\eta=0.01$",
            variable_label=r"$\rho$",
        )
        PlotPredictionPerVariable(
            self.base_x,
            pred_arr,
            rho_vals,
            title=r"Predicted polynomials (RMS_prop) $\eta=0.01$",
            n_epochs=self.n_epochs,
            target_func=self.target_func,
            variable_label=r"$\rho$",
        )
        plotThetas(
            theta_arr,
            rho_vals,
            title=r"Model parameters (RMS_prop) $\eta=0.01$",
            true_theta=self.true_theta,
            variable_label=r"$\rho$",
            variable_type="linear",
        )

        n_rho = 75
        n_eta = 75
        heat_rho = np.arctan(np.linspace(0.1, 10, n_rho))
        heat_rho = heat_rho * 2 / 3
        heat_eta = np.logspace(-3, -1, n_eta)

        schedulers = [[RMS_prop(eta, rho) for eta in heat_eta] for rho in heat_rho]
        error_arr = self.error_per_variables(schedulers, dim)

        df = pd.DataFrame(
            error_arr,
            index=[f"{rho:.2f}" for rho in heat_rho],
            columns=[f"{eta:.2e}" for eta in heat_eta],
        )
        plotHeatmap(df, title=f"Error after {self.n_epochs} epochs")

    def minibatch_analysis(self, dim: int = 2):
        # Flip so that the "better" args are plotted on top
        minibatches = np.flip(np.arange(1, 101))

        error_arr = np.zeros((len(minibatches), 150))
        theta_arr = np.zeros((len(minibatches), dim + 1))
        for i, batch_size in enumerate(minibatches):
            schedule = TimeDecay(1, 10, batch_size)
            Gradient = Gradients(
                self.n_points,
                self.x_vals,
                self.y_vals,
                self.model,
                self.method,
                schedule,
            )
            theta_arr[i, :] = Gradient.StochasticGradientDescent(
                self.base_theta, 150, batch_size
            ).ravel()
            error_arr[i, :] = Gradient.errors

        pred_arr = self.pred_per_theta(self.base_x, theta_arr, dim)

        PlotErrorPerVariable(
            error_arr,
            minibatches,
            title=r"Error per epoch TimeDecay ($t_0 = 1$, $t_1 = 10$)",
            variable_label="Minibatch size",
            variable_type="linear",
            colormap="viridis_r",
        )
        plotThetas(
            theta_arr,
            minibatches,
            title=r"Model parameters TimeDecay ($t_0 = 1$, $t_1 = 10$)",
            variable_label="Minibatch size",
            variable_type="linear",
            true_theta=self.true_theta,
        )
        PlotPredictionPerVariable(
            self.base_x,
            pred_arr,
            minibatches,
            title=r"Predicted polynomials TimeDecay ($t_0 = 1$, $t_1 = 10$)",
            n_epochs=150,
            target_func=self.target_func,
            variable_label="Minibatch size",
            variable_type="linear",
            colormap="viridis_r",
        )

        schedules = [
            Constant(0.01),
            Momentum(0.01, 0.9),
            Adagrad(0.01),
            AdagradMomentum(0.01, 0.9),
            Adam(0.01, 0.9, 0.999),
            RMS_prop(0.01, 0.9),
        ]
        schedule_names = [
            r"Constant ($\eta=0.01$)",
            r"Momentum ($\eta=0.01$, $\rho=0.9$)",
            r"Adagrad ($\eta=0.01$)",
            r"AdagradMomentum ($\eta=0.01$, $\rho=0.9$)",
            r"Adam ($\eta=0.01$, $\rho=0.9$, $\rho_2=0.999$)",
            r"RMS_prop ($\eta=0.01$, $\rho=0.9$)",
        ]
        epoch_size = [60, 60, 150, 60, 40, 60]
        for schedule, name, epoch in zip(schedules, schedule_names, epoch_size):
            error_arr, theta_arr = self.error_per_minibatch(
                schedule, minibatches, epoch
            )
            pred_arr = self.pred_per_theta(self.base_x, theta_arr, dim)

            PlotErrorPerVariable(
                error_arr,
                minibatches,
                title=f"Error per epoch {name}",
                variable_label="Minibatch size",
                variable_type="linear",
                colormap="viridis_r",
            )
            plotThetas(
                theta_arr,
                minibatches,
                title=f"Model parameters {name}",
                variable_label="Minibatch size",
                variable_type="linear",
                true_theta=self.true_theta,
            )
            PlotPredictionPerVariable(
                self.base_x,
                pred_arr,
                minibatches,
                title=f"Predicted polynomials {name}",
                n_epochs=epoch,
                target_func=self.target_func,
                variable_label="Minibatch size",
                variable_type="linear",
                colormap="viridis_r",
            )

    def gd_main(self):
        eta_num = 75
        eta_arr = np.logspace(-5, -1, eta_num)
        self.constant_analysis(eta_arr)

        rho_num = 75
        rho_arr = np.linspace(1 / self.n_points, 1, rho_num)
        self.momentum_analysis(eta_arr, rho_arr)

        # NOTE: Adagrad is less sensitive to the learning rate, so we can use larger values
        eta_arr = np.logspace(-3, 0, eta_num)
        self.adagrad_analysis(eta_arr)

        eta_arr = np.logspace(-5, 0, eta_num)
        rho_arr = np.linspace(1 / self.n_points, 0.99, rho_num)
        self.adagrad_momentum_analysis(eta_arr, rho_arr)

        rho2_num = 75
        rho2_arr = np.linspace(0.05, 0.999, rho2_num)
        self.adam_analysis(eta_arr, rho_arr, rho2_arr)

        self.rms_prop_analysis(eta_arr, rho_arr)
