import numpy as np
from Schedules import (
    Scheduler,
    Constant,
    Momentum,
    Adagrad,
    AdagradMomentum,
    Adam,
    RMS_prop,
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
import seaborn as sns
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


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
        error_arr = np.zeros((n_rho, n_eta))
        ynew = self.target_func(self.base_x)
        for i, rho in enumerate(heat_rho):
            for j, eta in enumerate(heat_eta):
                schedule = Momentum(eta, rho)
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

        fig, ax = plt.subplots()
        df = pd.DataFrame(
            error_arr,
            index=[f"{rho:.2f}" for rho in heat_rho],
            columns=[f"{eta:.2e}" for eta in heat_eta],
        )
        sns.heatmap(df, cmap="viridis", ax=ax)
        ax.set_xlabel(r"$\eta$")
        ax.set_ylabel(r"$\rho$")
        ax.set_title(f"Error after {self.n_epochs} epochs")
        plt.tight_layout()
        if savePlots:
            raise NotImplementedError
        if showPlots:
            plt.show()
        plt.close(fig)

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
        error_arr = np.zeros((n_rho, n_eta))
        ynew = self.target_func(self.base_x)
        for i, rho in enumerate(heat_rho):
            for j, eta in enumerate(heat_eta):
                schedule = AdagradMomentum(eta, rho)
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

        fig, ax = plt.subplots()
        df = pd.DataFrame(
            error_arr,
            index=[f"{rho:.2f}" for rho in heat_rho],
            columns=[f"{eta:.2e}" for eta in heat_eta],
        )
        sns.heatmap(df, cmap="viridis", ax=ax)
        ax.set_xlabel(r"$\eta$")
        ax.set_ylabel(r"$\rho$")
        ax.set_title(f"Error after {self.n_epochs} epochs")
        plt.tight_layout()
        if savePlots:
            raise NotImplementedError
        if showPlots:
            plt.show()
        plt.close(fig)

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
        error_arr = np.zeros((n_rho, n_eta))
        ynew = self.target_func(self.base_x)

        for i, rho in enumerate(heat_rho):
            for j, eta in enumerate(heat_eta):
                schedule = Adam(eta, rho, 0.999)
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

        fig, ax = plt.subplots()
        df = pd.DataFrame(
            error_arr,
            index=[f"{rho:.2f}" for rho in heat_rho],
            columns=[f"{eta:.2e}" for eta in heat_eta],
        )
        sns.heatmap(df, cmap="viridis", ax=ax)
        ax.set_xlabel(r"$\eta$")
        ax.set_ylabel(r"$\rho$")
        ax.set_title(f"Error after {self.n_epochs} epochs")
        plt.tight_layout()
        if savePlots:
            raise NotImplementedError
        if showPlots:
            plt.show()
        plt.close(fig)

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
        error_arr = np.zeros((n_rho, n_eta))
        ynew = self.target_func(self.base_x)

        for i, rho in enumerate(heat_rho):
            for j, eta in enumerate(heat_eta):
                schedule = RMS_prop(eta, rho)
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

        fig, ax = plt.subplots()
        df = pd.DataFrame(
            error_arr,
            index=[f"{rho:.2f}" for rho in heat_rho],
            columns=[f"{eta:.2e}" for eta in heat_eta],
        )
        sns.heatmap(df, cmap="viridis", ax=ax)
        ax.set_xlabel(r"$\eta$")
        ax.set_ylabel(r"$\rho$")
        ax.set_title(f"Error after {self.n_epochs} epochs")
        plt.tight_layout()
        if savePlots:
            raise NotImplementedError
        if showPlots:
            plt.show()
        plt.close(fig)
