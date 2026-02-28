from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy.stats import chi2, norm
from statsmodels.stats.multitest import multipletests
import pdb

# --- HELPER CLASSES ---
class LogisticModelFitter:
    """Handles fitting of logistic regression models."""
    
    def fit_model(self, df: pd.DataFrame, outcome: str, predictor: str, 
                  confounders: List[str]) -> Tuple[Dict[str, dict], float]:
        """Fit logistic model and extract coefficients plus interaction test."""
        clean_confounders = self._clean_confounders(confounders, predictor, df.columns)
        
        # Build and fit models
        multiplicative_formula = self._build_formula(outcome, predictor, clean_confounders, with_interaction=True)
        print(f"Fitting: {multiplicative_formula}")
        full_model = smf.logit(multiplicative_formula, data=df).fit(disp=0)
        
        additive_formula = self._build_formula(outcome, predictor, clean_confounders, with_interaction=False)
        reduced_model = smf.logit(additive_formula, data=df).fit(disp=0)
        
        # Extract results
        interaction_pval = self._likelihood_ratio_test(full_model, reduced_model)
        
        return full_model, interaction_pval
    
    def fit_stratified_model(self, df: pd.DataFrame, outcome: str, stratify_by: str, 
                           stratum_value, confounders: List[str]) -> Dict[str, dict]:
        """Fit logistic model for a specific stratum."""
        stratum_data = df[df[stratify_by] == stratum_value].copy()
        
        if len(stratum_data) == 0:
            raise ValueError(f"No data found for {stratify_by} = {stratum_value}")
        
        clean_confounders = self._clean_confounders(confounders, stratify_by, df.columns)
        formula = self._build_formula(outcome, "", clean_confounders, with_interaction=False)
        
        print(f"Fitting stratified model for {stratify_by} = {stratum_value}; {formula}")
        print(f"Sample size: {len(stratum_data)}")
        
        model = smf.logit(formula, data=stratum_data).fit(disp=0)
        odds_ratios = self._extract_exp_coefficients(model)
        return model, odds_ratios

    def fit_multivariate_model(self, df: pd.DataFrame, outcome: str, predictors: str, 
                  clean_confounders: List[str]) -> Tuple[Dict[str, dict], float]:
        """Fit logistic model and extract coefficients plus interaction test."""
        
        # Build and fit models
        interaction_str = ' + '.join([f'timebin * {p}' for p in predictors])
        covariate_str = ' + '.join(clean_confounders)
        interaction_formula = f"{outcome} ~ {interaction_str} + {covariate_str}"
        print(f"Formula:\n{interaction_formula}\n")
        model = smf.logit(interaction_formula, data=df).fit(disp=0)
        
        return model

    def _clean_confounders(self, confounders: List[str], predictor: str, 
                          available_cols: List[str]) -> List[str]:
        """Remove predictor and invalid columns from confounders."""
        valid_confounders = [c for c in confounders if c != predictor and c in available_cols]
        # Account for categorical vs continuous age
        if predictor == 'age_group':
            valid_confounders = [c for c in valid_confounders if c != 'age_firstLBP']
        return valid_confounders

    def _build_formula(self, outcome: str, predictor: str, confounders: List[str], 
                      with_interaction: bool = True) -> str:
        """Build logistic regression formula string."""
        confounder_str = " + " + " + ".join(confounders) if confounders else ""
        
        if with_interaction and predictor:
            return f"{outcome} ~ timebin * {predictor}{confounder_str}"
        elif predictor:
            return f"{outcome} ~ timebin + {predictor}{confounder_str}"
        else:
            return f"{outcome} ~ timebin{confounder_str}"

    def _likelihood_ratio_test(self, full_model, reduced_model) -> float:
        """Perform likelihood ratio test for interaction."""
        lr_stat = 2 * (full_model.llf - reduced_model.llf)
        df_diff = full_model.df_model - reduced_model.df_model
        p_val = chi2.sf(lr_stat, df=df_diff)
        print(f"Interaction LR test p-value: {p_val:.4f}")
        return p_val

    def _extract_exp_coefficients(self, model) -> Dict[str, dict]:
        """Extract all model coefficients into clean structure."""
        conf_int = model.conf_int()
        return {
            param: {
                "metric": np.exp(model.params[param]),
                "p_value": model.pvalues[param],
                "ci_lower": np.exp(conf_int.loc[param, 0]),
                "ci_upper": np.exp(conf_int.loc[param, 1])
            }
            for param in model.params.index
        }
    
class InteractionMetrics:
    """Handles calculation of interaction metrics like RERI and ROR."""
    
    def extract_interaction_terms(self, model, predictor: str, df: pd.DataFrame) -> Dict[str, dict]:
        """Extract ROR (interaction terms) and RERI coefficients using delta method."""
        interaction_coeffs = {}
        
        # Check if binary variable
        if df[predictor].dtype.name == 'category':
            levels = df[predictor].cat.categories.tolist()
        else:
            levels = sorted(df[predictor].unique())
        is_binary = (levels == [0.0, 1.0] or levels == [False, True])
        
        # Extract ROR and RERI for each timebin
        for timebin in ['B', 'C']:
            if is_binary:
                self._extract_binary_interaction_terms(
                    model, timebin, predictor, interaction_coeffs
                )
            else:
                self._extract_categorical_interaction_terms(
                    model, timebin, predictor, levels, interaction_coeffs
                )
        
        return interaction_coeffs

    def extract_interaction_odds(self, model, predictor: str, df: pd.DataFrame) -> Dict[str, dict]:
        """Calculate combined effects for interaction terms."""
        combined_ORs = {}
        
        # Check if binary variable
        if df[predictor].dtype.name == 'category':
            levels = df[predictor].cat.categories.tolist()
        else:
            levels = sorted(df[predictor].unique())
        is_binary = (levels == [0.0, 1.0] or levels == [False, True])
        
        # For each single exposure (predictor only, time bin only), calculate effects
        self._calculate_single_exposure_odds_ratios(model, predictor, combined_ORs)
        
        # For each timebin, calculate combined effects
        for timebin in ['B', 'C']:
            
            if is_binary:
                self._calculate_binary_multiexposure_odds_ratios(
                    model, timebin, predictor, combined_ORs
                )
            else:
                self._calculate_categorical_multiexposure_odds_ratios(
                    model, timebin, predictor, levels, combined_ORs
                )
        
        return combined_ORs

    def _extract_binary_interaction_terms(self, model, timebin: str, predictor: str, 
                                        interaction_terms: Dict[str, dict]):
        """Extract interaction terms for binary predictor."""
        interaction_param = f'timebin[T.{timebin}]:{predictor}'
        
        # Extract ROR (interaction term directly)
        if interaction_param in model.params.index:
            conf_int = model.conf_int()
            interaction_terms[f'ROR_{interaction_param}'] = {
                "metric": np.exp(model.params[interaction_param]),
                "p_value": model.pvalues[interaction_param],
                "ci_lower": np.exp(conf_int.loc[interaction_param, 0]),
                "ci_upper": np.exp(conf_int.loc[interaction_param, 1])
            }
        
        # Extract RERI using delta method
        reri_result = self._calculate_reri_delta_method(model, timebin, predictor)
        interaction_terms[f'RERI_{interaction_param}'] = reri_result

    def _extract_categorical_interaction_terms(self, model, timebin: str, predictor: str, 
                                             levels: List, interaction_terms: Dict[str, dict]):
        """Extract interaction terms for categorical predictor."""
        for level in levels[1:]:  # Skip reference level
            interaction_param = f'timebin[T.{timebin}]:{predictor}[T.{level}]'
            
            # Extract ROR (interaction term directly)
            if interaction_param in model.params.index:
                conf_int = model.conf_int()
                interaction_terms[f'ROR_{interaction_param}'] = {
                    "metric": np.exp(model.params[interaction_param]),
                    "p_value": model.pvalues[interaction_param],
                    "ci_lower": np.exp(conf_int.loc[interaction_param, 0]),
                    "ci_upper": np.exp(conf_int.loc[interaction_param, 1])
                }
            
            # Extract RERI using delta method
            reri_result = self._calculate_reri_delta_method(model, timebin, predictor, level)
            interaction_terms[f'RERI_{interaction_param}'] = reri_result

    def _calculate_reri_delta_method(self, model, timebin: str, predictor: str, 
                                   level: str = None) -> Dict[str, float]:
        """Calculate RERI using delta method for proper confidence intervals."""
        # Define parameter names based on predictor type
        if level:  # Categorical predictor
            main_param = f"{predictor}[T.{level}]"
            interaction_param = f"timebin[T.{timebin}]:{predictor}[T.{level}]"
        else:  # Binary predictor
            main_param = predictor
            interaction_param = f"timebin[T.{timebin}]:{predictor}"
            
        timebin_param = f'timebin[T.{timebin}]'
        
        # Check if all required parameters exist in model
        required_params = [main_param, timebin_param, interaction_param]
        if not all(param in model.params.index for param in required_params):
            return {"coefficient": np.nan, "p_value": np.nan, "ci_lower": np.nan, "ci_upper": np.nan}
        
        # Extract log-coefficients (betas)
        beta_main = model.params[main_param]
        beta_timebin = model.params[timebin_param]  
        beta_interaction = model.params[interaction_param]
        
        # Calculate RERI point estimate
        or_combined = np.exp(beta_main + beta_timebin + beta_interaction)
        or_main = np.exp(beta_main)
        or_timebin = np.exp(beta_timebin)
        
        reri_value = or_combined - or_main - or_timebin + 1
        
        # Calculate partial derivatives for delta method
        grad_main = or_combined - or_main
        grad_timebin = or_combined - or_timebin  
        grad_interaction = or_combined
        
        # Create gradient vector
        gradient = np.array([grad_main, grad_timebin, grad_interaction])
        
        # Extract relevant submatrix of variance-covariance matrix
        vcov = model.cov_params()
        param_indices = [main_param, timebin_param, interaction_param]
        vcov_submatrix = vcov.loc[param_indices, param_indices].values
        
        # Calculate variance using delta method
        reri_variance = gradient.T @ vcov_submatrix @ gradient
        reri_se = np.sqrt(reri_variance)
        
        # Calculate confidence intervals
        ci_lower = reri_value - 1.96 * reri_se
        ci_upper = reri_value + 1.96 * reri_se
        
        # Calculate z-score and p-value (Wald test: H0: RERI = 0)
        z_score = reri_value / reri_se if reri_se > 0 else np.nan
        p_value = 2 * (1 - norm.cdf(abs(z_score))) if not np.isnan(z_score) else np.nan
        
        return {
            "metric": reri_value,
            "p_value": p_value,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "standard_error": reri_se
        }
    
    def _calculate_single_exposure_odds_ratios(self, model, predictor: str, 
                                               combined_ORs: Dict[str, dict]) -> Dict[str, dict]:
        """Extract all model coefficients into clean structure."""
        conf_int = model.conf_int()
        for param in model.params.index:
            if (":" not in param) and any(substring in param for substring in [predictor, 'T.B', 'T.C']):
                combined_ORs[f'OR_{param}'] = {
                "metric": np.exp(model.params[param]),
                "p_value": model.pvalues[param],
                "ci_lower": np.exp(conf_int.loc[param, 0]),
                "ci_upper": np.exp(conf_int.loc[param, 1])
            }
        return

    def _calculate_binary_multiexposure_odds_ratios(self, model, timebin: str, predictor: str, 
                                        combined_ORs: Dict[str, dict]):
        """Calculate combined effect for binary predictor."""
        main_param = predictor
        timebin_param = f'timebin[T.{timebin}]'
        interaction_param = f'timebin[T.{timebin}]:{predictor}'
        
        required_params = [main_param, timebin_param, interaction_param]
        if all(param in model.params.index for param in required_params):
            combined_OR = self._calculate_multiexposure_odds_ratios(model, required_params)
            combined_ORs[f'OR_{interaction_param}'] = combined_OR

    def _calculate_categorical_multiexposure_odds_ratios(self, model, timebin: str, predictor: str, 
                                              levels: List, combined_ORs: Dict[str, dict]):
        """Calculate combined effects for categorical predictor."""
        for level in levels[1:]:  # Skip reference level
            main_param = f'{predictor}[T.{level}]'
            timebin_param = f'timebin[T.{timebin}]'
            interaction_param = f'timebin[T.{timebin}]:{predictor}[T.{level}]'
            
            required_params = [main_param, timebin_param, interaction_param]
            if all(param in model.params.index for param in required_params):
                combined_OR = self._calculate_multiexposure_odds_ratios(model, required_params)
                combined_ORs[f'OR_{interaction_param}'] = combined_OR

    def _calculate_multiexposure_odds_ratios(self, model, param_names: List[str]) -> Dict[str, float]:
        """Calculate combined coefficient, CI, and p-value for a linear combination of parameters."""
        # Get coefficients
        coeffs = [model.params[param] for param in param_names]
        combined_OR = sum(coeffs)

        # Calculate combined variance using variance-covariance matrix
        combined_var = 0
        n_params = len(param_names)
        
        # Add variances (diagonal terms)
        vcov = model.cov_params()
        for param in param_names:
            combined_var += vcov.loc[param, param]
        
        # Add covariances (off-diagonal terms, each pair twice)
        for i in range(n_params):
            for j in range(i+1, n_params):
                combined_var += 2 * vcov.loc[param_names[i], param_names[j]]
        
        # Calculate standard error
        combined_se = np.sqrt(combined_var)
        
        # Calculate z-score and p-value
        z_score = combined_OR / combined_se
        p_value = 2 * (1 - norm.cdf(abs(z_score)))  # Two-tailed test
        
        # Calculate confidence intervals (on log scale, then exponentiate)
        ci_lower_log = combined_OR - 1.96 * combined_se
        ci_upper_log = combined_OR + 1.96 * combined_se
        
        return {
            "metric": np.exp(combined_OR),
            "p_value": p_value,
            "ci_lower": np.exp(ci_lower_log),
            "ci_upper": np.exp(ci_upper_log),
            "log_metric": combined_OR,
            "standard_error": combined_se
        }

class ResultsFormatter:
    """Handles formatting of analysis results into tables."""
    
    def __init__(self, decimals: int = 4):
        self.decimals = decimals
    
    def create_stratified_table(self, df: pd.DataFrame, outcome: str, stratify_by: str, 
                               confounders: List[str], model_fitter) -> pd.DataFrame:
        """Create stratified analysis table for all labels of stratifying variable."""
        # Get labels in proper order
        if df[stratify_by].dtype.name == 'category':
            labels = df[stratify_by].cat.categories.tolist()
        else:
            labels = sorted(df[stratify_by].unique())
        
        predictor_rows = []
        # Create blank row corresponding to interaction p-value
        row = {
            'Variable': "NA",
            'Time A': "NA",
            'Time B': "NA",
            'Time C': "NA"
        }
        predictor_rows.append(row)
        
        for label in labels:
            try:
                # Fit model for this stratum
                _, odds_ratios = model_fitter.fit_stratified_model(df, outcome, stratify_by, label, confounders)
                
                # Create row for this stratum
                row = {
                    'Variable': f"within strata {stratify_by} {label}",
                    'Time A': "1",
                    'Time B': self._format_coeff(odds_ratios, 'timebin[T.B]'),
                    'Time C': self._format_coeff(odds_ratios, 'timebin[T.C]')
                }
                predictor_rows.append(row)
                
            except Exception as e:
                print(f"Error fitting model for {stratify_by} = {label}: {e}")
                # Add row with NA values
                row = {
                    'Variable': f"within strata {stratify_by} {label}",
                    'Time A': "NA",
                    'Time B': "NA", 
                    'Time C': "NA"
                }
                predictor_rows.append(row)
        
        return pd.DataFrame(predictor_rows)

    def create_results_summary(self, results_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create a summary dataframe with one row per coefficient/metric.
        
        Args:
            results_df: DataFrame with predictor analysis results containing 
                       'predictor', 'odds_ratios', and 'interaction_metrics' columns
            
        Returns:
            DataFrame with columns: predictor, key, metric, p_value, ci_lower, ci_upper
        """
        OR_summary_rows = []
        interaction_summary_rows = []
        for _, row in results_df.iterrows():
            predictor = row['predictor']
            
            # Extract odds ratios
            if 'odds_ratios' in row and row['odds_ratios']:
                for key, values in row['odds_ratios'].items():
                    OR_summary_rows.append({
                        'predictor': predictor,
                        'key': key,
                        'metric': values.get('metric', np.nan),
                        'ci_lower': values.get('ci_lower', np.nan),
                        'ci_upper': values.get('ci_upper', np.nan),
                        'p_value': values.get('p_value', np.nan)
                    })
            
            # Extract interaction metrics
            if 'interaction_metrics' in row and row['interaction_metrics']:
                for key, values in row['interaction_metrics'].items():
                    interaction_summary_rows.append({
                        'predictor': predictor,
                        'key': key,
                        'metric': values.get('metric', np.nan),  # Handle both naming conventions
                        'ci_lower': values.get('ci_lower', np.nan),
                        'ci_upper': values.get('ci_upper', np.nan),
                        'p_value': values.get('p_value', np.nan)
                    })
        
        return [pd.DataFrame(OR_summary_rows), pd.DataFrame(interaction_summary_rows)]

    def create_interaction_table(self, results_df: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction table with ROR and RERI columns."""
        all_rows = []
        for _, row in results_df.iterrows():
            predictor = row['predictor']
            interaction_terms = row['interaction_metrics']
            # interaction_pval = row['corrected_interaction_pval']
            if interaction_terms is None:
                return pd.DataFrame()
            
            # Get levels in proper order for categorical variables
            if df[predictor].dtype.name == 'category':
                levels = df[predictor].cat.categories.tolist()
            else:
                levels = sorted(df[predictor].unique())
            # Check if binary variable
            is_binary = (levels == [0.0, 1.0] or levels == [False, True])
            
            if is_binary:
                all_rows.append({
                    'Variable': predictor,                
                    'ROR Time B': self._format_coeff(interaction_terms, f"ROR_timebin[T.B]:{predictor}", self.decimals),
                    'ROR Time C': self._format_coeff(interaction_terms, f"ROR_timebin[T.C]:{predictor}", self.decimals),
                    'RERI Time B': self._format_coeff(interaction_terms, f"RERI_timebin[T.B]:{predictor}", self.decimals),
                    'RERI Time C': self._format_coeff(interaction_terms, f"RERI_timebin[T.C]:{predictor}", self.decimals)
                })
            else:
                for level in levels[1:]:
                    all_rows.append({
                        'Variable': f"{predictor} {level}",
                        'ROR Time B': self._format_coeff(interaction_terms, f"ROR_timebin[T.B]:{predictor}[T.{level}]", self.decimals),
                        'ROR Time C': self._format_coeff(interaction_terms, f"ROR_timebin[T.C]:{predictor}[T.{level}]", self.decimals),
                        'RERI Time B': self._format_coeff(interaction_terms, f"RERI_timebin[T.B]:{predictor}[T.{level}]", self.decimals),
                        'RERI Time C': self._format_coeff(interaction_terms, f"RERI_timebin[T.C]:{predictor}[T.{level}]", self.decimals)
                    })
        
        return pd.DataFrame(all_rows)
    
    def create_formatted_OR_summary(self, results_df: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Create clean intercepts table in requested format."""
        all_rows = []
        for _, row in results_df.iterrows():
            predictor = row['predictor']
            odds_ratios = row['odds_ratios']
            interaction_pval = row['corrected_interaction_pval']

            # Get levels in proper order for categorical variables
            if df[predictor].dtype.name == 'category':
                levels = df[predictor].cat.categories.tolist()
            else:
                levels = sorted(df[predictor].unique())
            # Check if binary variable
            is_binary = (levels == [0.0, 1.0] or levels == [False, True])

            predictor_rows = self._create_rows(predictor, levels, odds_ratios, interaction_pval, is_binary)
            all_rows.extend(predictor_rows)
        return pd.DataFrame(all_rows)

    def _create_rows(self, predictor: str, levels: List[str], 
                            odds_ratios: Dict[str, dict], interaction_pval: float, is_binary: bool = False) -> dict:
        """Create table row for reference category."""
        interaction_row = {
                    'Variable': f"{predictor} interaction",
                    'Time A': f"{self._format_pval(interaction_pval)}",
                    'Time B': "NA",
                    'Time C': "NA"
                    }
        
        predictor_rows = [interaction_row]

        if is_binary:
            reference_row = {
                'Variable': f"no {predictor}",
                'Time A': 1,
                'Time B': self._format_coeff(odds_ratios, 'OR_timebin[T.B]'),
                'Time C': self._format_coeff(odds_ratios, 'OR_timebin[T.C]')
            }
            predictor_rows.append(reference_row)
            # For binary variables, add the predictor level
            level_row = {
                'Variable': predictor,
                'Time A': self._format_coeff(odds_ratios, f"OR_{predictor}"),
                'Time B': self._format_coeff(odds_ratios, f"OR_timebin[T.B]:{predictor}"),
                'Time C': self._format_coeff(odds_ratios, f"OR_timebin[T.C]:{predictor}")
            }

            predictor_rows.append(level_row)

        else:
            reference_row = {
                'Variable': f"{predictor} {levels[0]}",
                'Time A': 1,
                'Time B': self._format_coeff(odds_ratios, 'OR_timebin[T.B]'),
                'Time C': self._format_coeff(odds_ratios, 'OR_timebin[T.C]')
            }
            predictor_rows.append(reference_row)
            # For categroical variables, add the predictor level
            for level in levels[1:]:
                level_row = {
                    'Variable': f"{predictor} {level}",
                    'Time A': self._format_coeff(odds_ratios, f"OR_{predictor}[T.{level}]"),
                    'Time B': self._format_coeff(odds_ratios, f"OR_timebin[T.B]:{predictor}[T.{level}]"),
                    'Time C': self._format_coeff(odds_ratios, f"OR_timebin[T.C]:{predictor}[T.{level}]")
                }

                predictor_rows.append(level_row)

        return predictor_rows

    def _format_pval(self, p_value) -> str:
        """Format p-value according to standard conventions."""
        if pd.isna(p_value):
            return "NA"
        elif p_value < 0.001:
            return "p < .001"
        elif p_value < 0.01:
            return f"p = {p_value:.3f}".replace("0.", ".")
        elif p_value < 0.99:
            return f"p = {p_value:.2f}".replace("0.", ".")
        else:
            return f"p > .99"

    def _format_coeff(self, metrics: Dict[str, dict], key: str, decimals: int = 2) -> str:
        """Safely format coefficient, returning 'NA' if not found."""
        # Check if key exists first
        if key not in metrics:
            return "NA"
        
        # Extract coefficient value
        coeff_value = metrics[key]["metric"]
        ci_lower_value = metrics[key]["ci_lower"]
        ci_upper_value = metrics[key]["ci_upper"]
        p_value = metrics[key]["p_value"]

        # Format if not NaN, otherwise return NA
        if np.isnan(coeff_value):
            return "NA"
        else:
            return f"{coeff_value:.{decimals}f} ({ci_lower_value:.{decimals}f}, {ci_upper_value:.{decimals}f}); {self._format_pval(p_value)}"


# --- MAIN CLASSES ---
# Main Analysis Orchestrator
class LogisticAnalysisOrchestrator:
    """Coordinates the entire logistic regression analysis workflow."""
    
    def __init__(self, decimals: int = 4):
        self.model_fitter = LogisticModelFitter()
        self.interaction_metrics = InteractionMetrics()
        self.formatter = ResultsFormatter(decimals)
        
    def fdr_correct(self, pvals, alpha: float = 0.05) -> pd.DataFrame:
        """Apply Benjamini-Hochberg FDR correction to interaction p-values."""
        rejected, pvals_corrected, _, _ = multipletests(pvals, alpha=alpha, method='fdr_bh')
        return pvals_corrected

    def bonferroni_correct(self, pvals, alpha: float = 0.05) -> pd.DataFrame:
        """Apply Bonferroni correction to interaction p-values."""
        rejected, pvals_corrected, _, _ = multipletests(pvals, alpha=alpha, method='bonferroni')
        return pvals_corrected

    def analyze_stratified_predictors(self, df: pd.DataFrame, predictors: List[str], 
                                    confounders: List[str], outcome: str = 'opioid') -> pd.DataFrame:
        """Analyze multiple predictors using stratified approach."""
        results_data = []
        
        for predictor in predictors:
            print(f"\n=== Stratified Analysis for {predictor} ===")
            try:
                table = self.formatter.create_stratified_table(
                    df, outcome, predictor, confounders, self.model_fitter
                )
                results_data.append(table)
            except Exception as e:
                print(f"Error in stratified analysis for {predictor}: {e}")
                continue
        
        return pd.concat(results_data, ignore_index=True) if results_data else pd.DataFrame()

    def analyze_predictors(self, df: pd.DataFrame, predictors: List[str], 
                          confounders: List[str], outcome: str = 'opioid') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Main function to analyze multiple predictors and return combined tables."""
        # Fit models
        results_data = []
        for predictor in predictors:
            print(f"\n=== Analyzing {predictor} ===")
            
            model, interaction_pval  = self.model_fitter.fit_model(
                df, outcome, predictor, confounders
            )
            
            odds_ratios = self.interaction_metrics.extract_interaction_odds(
                model, predictor, df
            )
            
            interaction_terms = self.interaction_metrics.extract_interaction_terms(
                model, predictor, df
            )
            
            results_data.append({
                'predictor': predictor,
                'interaction_pval': interaction_pval,
                'odds_ratios': odds_ratios,
                'interaction_metrics': interaction_terms
            })
        results_df = pd.DataFrame(results_data)

        # Apply multiple comparisons correction
        interaction_pvals = results_df['interaction_pval'].to_list()
        corrected_pvals = self.fdr_correct(interaction_pvals)
        results_df['corrected_interaction_pval'] = corrected_pvals
        OR_results, interaction_results = self.formatter.create_results_summary(results_df)
        # bonferroni and FRD identify the same significant interactions

        # Create tables
        formatted_OR_results = self.formatter.create_formatted_OR_summary(results_df, df)
        formatted_interaction_results = self.formatter.create_interaction_table(results_df, df)
        
        print('bonf')
        return OR_results, interaction_results, corrected_pvals, formatted_OR_results, formatted_interaction_results
    
    def analyze_multivariate_predictors(self, df: pd.DataFrame, predictors: List[str], 
                          confounders: List[str], outcome: str = 'opioid') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Main function to analyze multiple predictors and return combined tables."""

        print(f"\n=== Analyzing {predictors} ===")
        
        model  = self.model_fitter.fit_multivariate_model(
            df, outcome, predictors, confounders
        )

        results_data = []
        for predictor in predictors:
            
            interaction_terms = self.interaction_metrics.extract_interaction_terms(
                model, predictor, df
            )
            
            results_data.append({
                'predictor': predictor,
                'interaction_pval': None,
                'odds_ratios': None,
                'interaction_metrics': interaction_terms
            })

        results_df = pd.DataFrame(results_data)

        # Create tables
        OR_results, interaction_results = self.formatter.create_results_summary(results_df)
        formatted_interaction_results = self.formatter.create_interaction_table(results_df, df)
        
        return interaction_results, formatted_interaction_results
    