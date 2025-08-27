<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->
- [x] Verify that the copilot-instructions.md file in the .github directory is created.

- [x] Clarify Project Requirements
	<!-- Ask for project type, language, and frameworks if not specified. Skip if already provided. -->

- [x] Scaffold the Project
	<!--
	Ensure that the previous step has been marked as completed.
	Call project setup tool with projectType parameter.
	Run scaffolding command to create project files and folders.
	Use '.' as the working directory.
	If no appropriate projectType is available, search documentation using available tools.
	Otherwise, create the project structure manually using appropriate file creation tools.
	-->

- [ ] Customize the Project
	<!--
	Verify that all previous steps have been completed successfully and you have marked the step as completed.
	Develop a plan to modify codebase according to user requirements.
	Apply modifications using appropriate tools and user-provided references.
	Skip this step for "Hello World" projects.
	-->

- [ ] Install Required Extensions
	<!-- ONLY install extensions provided mentioned in the get_project_setup_info. Skip this step otherwise and mark as completed. -->

- [ ] Compile the Project
	<!--
	Verify that all previous steps have been completed.
	Install any missing dependencies.
	Run diagnostics and resolve any issues.
	Check for markdown files in project folder for relevant instructions on how to do this.
	-->

- [ ] Create and Run Task
	<!--
	Verify that all previous steps have been completed.
	Check https://code.visualstudio.com/docs/debugtest/tasks to determine if the project needs a task. If so, use the create_and_run_task to create and launch a task based on package.json, README.md, and project structure.
	Skip this step otherwise.
	 -->

- [ ] Launch the Project
	<!--
	Verify that all previous steps have been completed.
	Prompt user for debug mode, launch only if confirmed.
	 -->

- [ ] Ensure Documentation is Complete
	<!--
	Verify that all previous steps have been completed.
	Verify that README.md and the copilot-instructions.md file in the .github directory exists and contains current project information.
	Clean up the copilot-instructions.md file in the .github directory by removing all HTML comments.
	 -->


# Copilot Instructions — CVXJAX

**Purpose:** Guide GitHub Copilot to scaffold, customize, and validate the **CVXJAX** repository — a JAX-native convex optimization library (MVP analogue of CVXPY) with LP/QP modeling, a dense IPM solver, an OSQP bridge via `jaxopt`, and implicit differentiation through the KKT system.

> **Stack:** Python 3.10–3.11, JAX/JAXLIB, JAXOPT, NumPy, SciPy, PyTest, Ruff, pre-commit.  
> **Target OS:** Linux/macOS dev; CI on Ubuntu.

---

## ✅ Checklist & Step-by-Step Plan

- [x] Verify that the `copilot-instructions.md` file in the `.github` directory is created.

- [ ] Clarify Project Requirements  
  - Project: **CVXJAX** (Python package).  
  - Language: **Python** (typed), JAX-friendly code.  
  - Frameworks/libraries: `jax`, `jaxlib`, `jaxopt`, `numpy`, `scipy`, `pytest`, `pytest-cov`, `typing_extensions`, `ruff`, `pre-commit`.  
  - Minimum Python: 3.10.

- [ ] Scaffold the Project  
  - Create this exact structure:
    ```
    cvxjax/
      cvxjax/
        __init__.py
        api.py
        expressions.py
        atoms.py
        constraints.py
        canonicalize.py
        diff.py
        utils/
          __init__.py
          shapes.py
          scaling.py
          checking.py
        solvers/
          __init__.py
          ipm_qp.py
          osqp_bridge.py
      examples/
        quickstart_qp.py
        lasso_training_loop.py
        portfolio_qp.py
      tests/
        test_api.py
        test_canonicalize.py
        test_solver_ipm.py
        test_solver_osqp.py
        test_diff.py
      docs/
        index.md
        quickstart.md
        concepts.md
        troubleshooting.md
      pyproject.toml
      README.md
      LICENSE
      .gitignore
      .pre-commit-config.yaml
      .ruff.toml
      .github/workflows/ci.yml
      Makefile
    ```
  - Packaging: `pyproject.toml` with `build-system` = hatchling or uv; project name `cvxjax`, version `0.1.0`.  
  - Add `LICENSE` = Apache-2.0.  
  - Add `.gitignore` for Python/jupyter/venv.

- [ ] Customize the Project  
  - Fill **stubs with compiling code** and **docstrings**:
    - `api.py`: `Variable`, `Parameter`, `Constant`, `Minimize`, `Maximize`, `Constraint`, `Problem`, `Solution`.  
    - `expressions.py`: expression graph with metadata; operator overloading for arithmetic and comparisons.  
    - `atoms.py`: `sum_squares`, `quad_form(x, Q)`, `abs`, `square`, `matmul`, `sum`, `reshape`, `slice`.  
    - `constraints.py`: inequality/equality + `box(x, lb, ub)`.  
    - `canonicalize.py`: dense LP/QP packing.  
    - `solvers/ipm_qp.py`: primal-dual Mehrotra IPM with KKT solve, line search, stopping criteria.  
    - `solvers/osqp_bridge.py`: adapter to `jaxopt.OSQP`.  
    - `diff.py`: `custom_vjp` for QP solve with implicit differentiation; gradcheck helper.  
    - `utils/`: shapes, scaling, PSD and feasibility checks.  
  - Add runnable **examples** and **docs**.

- [ ] Install Required Extensions  
  - If none are specified, mark complete.

- [ ] Compile the Project  
  - Create and activate venv.  
  - `pip install -e .[dev]` (dev extras include pytest, pytest-cov, ruff, pre-commit).  
  - Run `pre-commit install`.  
  - Run `pytest -q --maxfail=1 --disable-warnings`.  
  - Run `ruff check .`.

- [ ] Create and Run Task  
  - If tasks are needed, create VS Code tasks for `pytest` and `ruff`.  
  - Otherwise, mark complete.

- [ ] Launch the Project  
  - Run `examples/quickstart_qp.py`.  
  - Optional: launch in debug mode if requested.

- [ ] Ensure Documentation is Complete  
  - Confirm `README.md` and `.github/copilot-instructions.md` exist and are up to date.  
  - Ensure `README.md` quickstart matches `examples/quickstart_qp.py`.  
  - Remove stale references.  

---

## Copilot Task Prompts

**Scaffold the repository**
- “Create the CVXJAX repo with the exact structure listed in the checklist. Populate each file with compiling stubs and Google-style docstrings. Use `pyproject.toml` with hatchling, project name `cvxjax`, version `0.1.0`, Python `>=3.10`.”

**Fill core API and expressions**
- “In `cvxjax/api.py`, implement `Variable`, `Parameter`, `Constant`, `Minimize`, `Maximize`, `Constraint`, `Problem`, and `Solution`. Make objects pytrees where appropriate. `Problem.solve()` should call canonicalization and then `solvers.ipm_qp.solve_qp_dense`. `Problem.solve_jit()` should return a compiled path with static shapes.”

**Atoms and canonicalization**
- “In `cvxjax/atoms.py`, add `sum_squares`, `quad_form(x, Q)`, `abs`, `square`, `matmul`, `sum`, `reshape`, `slice`.”  
- “In `cvxjax/canonicalize.py`, implement dense QP/LP canonicalization, normalize equalities/inequalities, return dataclasses with static shapes.”

**Solvers**
- “In `cvxjax/solvers/ipm_qp.py`, implement a primal-dual Mehrotra IPM with predictor-corrector, KKT assembly, ridge stabilization, fraction-to-boundary step, residual-based stopping, return `Solution` with residual traces.”  
- “In `cvxjax/solvers/osqp_bridge.py`, connect to `jaxopt.OSQP`, map data, unify results.”

**Differentiation**
- “In `cvxjax/diff.py`, wrap solve in `custom_vjp`, implement implicit differentiation of KKT with ridge stabilization, add `gradcheck_qp` helper.”

**Examples and docs**
- “Populate `examples/quickstart_qp.py` with a 3-var QP, solve and print solution, then show `jax.grad` through `solve_jit`.”  
- “Populate `examples/lasso_training_loop.py` (lasso as QP) and `examples/portfolio_qp.py` (box-bounded budget).”  
- “Write `docs/index.md`, `docs/quickstart.md`, `docs/concepts.md`, `docs/troubleshooting.md` aligned with examples.”

**Tests**
- “Add tests:
  - `test_api.py`: small QP with equality+inequality, assert KKT residuals < 1e-6.
  - `test_canonicalize.py`: shape/packing invariants.
  - `test_solver_ipm.py`: random PD Q, check convergence.
  - `test_solver_osqp.py`: compare OSQP bridge results.
  - `test_diff.py`: gradient check via finite differences.”

**Tooling and CI**
- “Add `.ruff.toml`, `.pre-commit-config.yaml`, and `Makefile`.  
- `ci.yml`: run on Ubuntu, Python 3.10 & 3.11, install JAX CPU wheels, run Ruff + PyTest.”

---

## Runbook

**Create venv and install**
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .[dev]
pre-commit install


