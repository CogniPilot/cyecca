"""Hierarchical model composition functionality.

Provides tools for building complex systems by composing simpler subsystems
with signal routing between components.
"""

import copy
from dataclasses import field as dc_field
from dataclasses import fields, make_dataclass

import casadi as ca

from .decorators import compose_states, symbolic

__all__ = [
    "CompositionMixin",
    "SignalRef",
    "SubmodelProxy",
]


class SignalRef:
    """Reference to a signal in a model for autocomplete-friendly connections."""

    def __init__(self, model_name: str, signal_name: str):
        self.model_name = model_name
        self.signal_name = signal_name
        self._path = f"{model_name}.{signal_name}"

    def __str__(self):
        return self._path

    def __repr__(self):
        return f"SignalRef('{self._path}')"


class SubmodelProxy:
    """Proxy object that allows attribute access to submodel signals."""

    def __init__(self, name: str, model):
        self._name = name
        self._model = model

    def __getattr__(self, attr: str):
        # Check if attribute exists in submodel's state, input, or output
        if hasattr(self._model, "x") and hasattr(self._model.x, attr):
            return SignalRef(self._name, attr)
        elif hasattr(self._model, "u") and hasattr(self._model.u, attr):
            return SignalRef(self._name, attr)
        elif hasattr(self._model, "y") and hasattr(self._model.y, attr):
            return SignalRef(self._name, attr)
        else:
            raise AttributeError(f"Signal '{attr}' not found in submodel '{self._name}'")


class CompositionMixin:
    """Mixin providing hierarchical composition methods for ModelSX/ModelMX.

    This mixin adds:
    - compose() classmethod for creating parent models
    - add_submodel() for adding subsystems
    - connect() for routing signals between submodels
    - build_composed() for building the unified dynamics
    """

    @classmethod
    def compose(
        cls,
        submodels: dict[str, "CompositionMixin"],
        state_type=None,
        input_type=None,
        param_type=None,
        output_type=None,
    ):
        """Compose multiple submodels into a single parent model.

        Creates a parent model with submodels already attached, ready for connections.
        Automatically composes states from all submodels if not specified.

        Args:
            submodels: Dictionary of {name: model} pairs
            state_type: Parent state type (auto-composed from submodels if None)
            input_type: Parent input type (empty if None)
            param_type: Parent parameter type (empty if None)
            output_type: Parent output type (empty if None)

        Returns:
            Parent model with submodels attached as attributes

        Example:
            >>> plant = sportcub()  # doctest: +SKIP
            >>> controller = autolevel_controller()  # doctest: +SKIP
            >>> parent = ModelSX.compose({"plant": plant, "controller": controller})  # doctest: +SKIP
            >>> parent.connect(controller.u.q, plant.x.r)  # doctest: +SKIP
        """
        # Auto-compose state type if not provided
        if state_type is None:
            state_types = [sub.state_type for sub in submodels.values()]
            state_type = compose_states(*state_types)

        # Create empty types if not provided
        if input_type is None:

            @symbolic
            class EmptyInputs:
                pass

            input_type = EmptyInputs

        if param_type is None:

            @symbolic
            class EmptyParams:
                pass

            param_type = EmptyParams

        if output_type is None:

            @symbolic
            class EmptyOutputs:
                pass

            output_type = EmptyOutputs

        # Create parent model
        parent = cls.create(state_type, input_type, param_type, output_type)

        # Add all submodels and create proxies for connection API
        for name, submodel in submodels.items():
            parent.add_submodel(name, submodel)

        return parent

    def add_submodel(
        self,
        name: str,
        submodel,
        state_connections: dict[str, str] = None,
        input_connections: dict[str, str] = None,
        output_connections: dict[str, str] = None,
    ):
        """Add a submodel for hierarchical composition.

        This allows building complex systems from simpler components. Submodels
        are stored and their states/inputs/outputs can be connected to the parent
        model or other submodels.

        Args:
            name: Unique identifier for this submodel
            submodel: ModelSX instance to add as a component
            state_connections: Map submodel states to parent states
            input_connections: Map submodel inputs to parent inputs or submodel outputs
            output_connections: Map submodel outputs to parent outputs

        Example:
            >>> parent.add_submodel("aircraft", aircraft,  # doctest: +SKIP
            ...     input_connections={"aircraft.thr": "controller.thr"})
        """
        if not hasattr(self, "_submodels"):
            self._submodels = {}
            self._state_connections = {}
            self._input_connections = {}
            self._output_connections = {}

        if name in self._submodels:
            raise ValueError(f"Submodel '{name}' already exists")

        self._submodels[name] = submodel

        # Create proxy for autocomplete-friendly signal access
        setattr(self, name, SubmodelProxy(name, submodel))

        if state_connections:
            self._state_connections[name] = state_connections
        if input_connections:
            self._input_connections[name] = input_connections
        if output_connections:
            self._output_connections[name] = output_connections

    def connect(self, target, source):
        """Add a connection between signals.

        Accepts either string paths or SignalRef objects for autocomplete support.

        Args:
            target: Target signal (SignalRef or string like "controller.u.q")
            source: Source signal (SignalRef or string like "plant.x.r")

        Example:
            >>> parent.connect("controller.u.q", "plant.x.r")  # doctest: +SKIP
            >>> parent.connect(parent.controller.u.q, parent.plant.x.r)  # doctest: +SKIP
        """

        # Helper to convert signal references to path strings
        def to_path_string(sig):
            if isinstance(sig, SignalRef):
                return str(sig)
            elif isinstance(sig, (ca.SX, ca.MX, ca.DM)):
                raise TypeError(
                    f"Cannot connect CasADi symbol directly: {sig}. "
                    f"Use string paths like 'controller.u.q' or SignalRef"
                )
            else:
                return sig

        target_str = to_path_string(target)
        source_str = to_path_string(source)

        if not hasattr(self, "_submodels"):
            self._submodels = {}
            self._state_connections = {}
            self._input_connections = {}
            self._output_connections = {}

        target_parts = target_str.split(".")
        source_parts = source_str.split(".")

        if len(target_parts) < 2:
            raise ValueError(f"Invalid target format: {target_str}. Expected 'model.type.field'")

        target_model = target_parts[0]

        # Determine connection type
        if target_model == "y":
            # Output connection
            source_model = source_parts[0]
            if source_model not in self._output_connections:
                self._output_connections[source_model] = {}
            self._output_connections[source_model][source_str] = target_str
        elif target_model == "x":
            # State connection
            source_model = source_parts[0]
            if source_model not in self._state_connections:
                self._state_connections[source_model] = {}
            self._state_connections[source_model][source_str] = target_str
        else:
            # Input connection
            if target_model not in self._input_connections:
                self._input_connections[target_model] = {}
            self._input_connections[target_model][target_str] = source_str

    def get_connections(self, target_model: str = None):
        """Get input connections for a submodel.
        
        Args:
            target_model: Name of the target submodel. If None, returns all connections.
            
        Returns:
            dict: Dictionary mapping target signals to source signals.
                  If target_model is None, returns nested dict {model: {target: source}}.
                  
        Example:
            >>> parent.connect("child2.u.u", "child1.y.y")  # doctest: +SKIP
            >>> parent.get_connections("child2")  # doctest: +SKIP
            {'child2.u.u': 'child1.y.y'}
        """
        if not hasattr(self, "_input_connections"):
            return {} if target_model is None else {}
        
        if target_model is None:
            return dict(self._input_connections)
        
        return dict(self._input_connections.get(target_model, {}))

    def build_composed(self, integrator: str = "rk4", integrator_options: dict = None):
        """Build a composed model from added submodels.

        Creates unified dynamics integrating all submodels with signal routing.

        Example:
            >>> parent.add_submodel("aircraft", aircraft, ...)  # doctest: +SKIP
            >>> parent.add_submodel("controller", controller, ...)  # doctest: +SKIP
            >>> parent.build_composed(integrator="rk4")  # doctest: +SKIP
        """
        if not hasattr(self, "_submodels") or not self._submodels:
            raise ValueError("No submodels added. Use add_submodel() first.")

        if integrator_options is None:
            integrator_options = {}

        # Build index maps for parent model
        self._build_index_maps()

        # Build combined state vector from all submodels
        submodel_state_slices = {}
        offset = 0
        for name, submodel in self._submodels.items():
            n_states = submodel.x.size1()
            submodel_state_slices[name] = (offset, offset + n_states)
            offset += n_states

        total_states = offset

        # Create combined dynamics function
        x_combined = ca.SX.sym("x_combined", total_states)
        u_parent = self.u.as_vec()
        p_parent = self.p.as_vec()

        # Extract submodel states from combined vector
        submodel_states = {}
        for name, (start, end) in submodel_state_slices.items():
            submodel_states[name] = x_combined[start:end]

        # Build input vectors for each submodel by resolving connections
        submodel_inputs = {}
        submodel_outputs = {}

        # First pass: Build preliminary inputs with state and parent input connections resolved
        for name, submodel in self._submodels.items():
            u_sub_dict = {}
            input_conns = self._input_connections.get(name, {})

            for field_obj in fields(submodel.u):
                field_name = field_obj.name
                full_path = f"{name}.u.{field_name}"

                if full_path in input_conns:
                    source = input_conns[full_path]

                    if source.startswith("u."):
                        # Parent input connection
                        parent_field = source[2:]
                        if hasattr(self.u, parent_field):
                            u_sub_dict[field_name] = getattr(self.u, parent_field)
                        else:
                            u_sub_dict[field_name] = getattr(submodel.u0, field_name)
                    else:
                        parts = source.split(".", 2)
                        if len(parts) >= 3 and parts[1] == "x":
                            # State connection
                            source_model, source_type, source_field = parts
                            if source_model in submodel_states:
                                u_sub_dict[field_name] = self._extract_field_from_vec(
                                    submodel_states[source_model],
                                    self._submodels[source_model].x,
                                    source_field,
                                )
                            else:
                                u_sub_dict[field_name] = getattr(submodel.u0, field_name)
                        else:
                            # Output connection - can't resolve yet, use default
                            u_sub_dict[field_name] = getattr(submodel.u0, field_name)
                else:
                    u_sub_dict[field_name] = getattr(submodel.u0, field_name)

            u_sub_list = [u_sub_dict[f.name] for f in fields(submodel.u)]
            submodel_inputs[name] = ca.vertcat(*u_sub_list) if u_sub_list else ca.DM.zeros(0, 1)

        # Evaluate all submodel outputs with preliminary inputs
        for name, submodel in self._submodels.items():
            if hasattr(submodel, "f_y"):
                x_sub = submodel_states[name]
                u_sub = submodel_inputs[name]
                p_sub = submodel.p0.as_vec()
                y_sub = submodel.f_y(x_sub, u_sub, p_sub)
                submodel_outputs[name] = y_sub

        # Second pass: Resolve output connections
        for name, submodel in self._submodels.items():
            u_sub_dict = {}
            input_conns = self._input_connections.get(name, {})

            for field_obj in fields(submodel.u):
                field_name = field_obj.name
                full_path = f"{name}.u.{field_name}"

                if full_path in input_conns:
                    source = input_conns[full_path]

                    if source.startswith("u."):
                        # Parent input
                        parent_field = source[2:]
                        if hasattr(self.u, parent_field):
                            u_sub_dict[field_name] = getattr(self.u, parent_field)
                        else:
                            u_sub_dict[field_name] = getattr(submodel.u0, field_name)

                    elif "." in source:
                        parts = source.split(".", 2)
                        if len(parts) >= 3:
                            source_model, source_type, source_field = parts

                            if source_type == "y" and source_model in submodel_outputs:
                                u_sub_dict[field_name] = self._extract_field_from_vec(
                                    submodel_outputs[source_model],
                                    self._submodels[source_model].y,
                                    source_field,
                                )
                            elif source_type == "x" and source_model in submodel_states:
                                u_sub_dict[field_name] = self._extract_field_from_vec(
                                    submodel_states[source_model],
                                    self._submodels[source_model].x,
                                    source_field,
                                )
                            else:
                                u_sub_dict[field_name] = getattr(submodel.u0, field_name)
                        else:
                            u_sub_dict[field_name] = getattr(submodel.u0, field_name)
                    else:
                        u_sub_dict[field_name] = getattr(submodel.u0, field_name)
                else:
                    u_sub_dict[field_name] = getattr(submodel.u0, field_name)

            u_sub_list = [u_sub_dict[f.name] for f in fields(submodel.u)]
            submodel_inputs[name] = ca.vertcat(*u_sub_list) if u_sub_list else ca.DM.zeros(0, 1)

        # Evaluate each submodel's dynamics with resolved inputs
        f_x_parts = []
        for name, submodel in self._submodels.items():
            x_sub = submodel_states[name]
            u_sub = submodel_inputs[name]
            p_sub = submodel.p0.as_vec()
            dx_sub = submodel.f_x(x_sub, u_sub, p_sub)
            f_x_parts.append(dx_sub)

        # Combine all submodel derivatives
        f_x_combined = ca.vertcat(*f_x_parts)

        # Create composed dynamics function
        self.f_x = ca.Function(
            "f_x_composed",
            [x_combined, u_parent, p_parent],
            [f_x_combined],
            ["x", "u", "p"],
            ["dx_dt"],
        )

        # Build composed output function if parent has outputs
        if self.output_type is not None and hasattr(self, "y"):
            self._build_composed_outputs(submodel_state_slices, submodel_outputs)

        # Store composition metadata
        self._composed = True
        self._submodel_state_slices = submodel_state_slices
        self._total_composed_states = total_states

        # Build integrator
        if integrator == "rk4":
            self._build_rk4_integrator(integrator_options)
        elif integrator == "euler":
            self._build_euler_integrator()
        else:
            raise ValueError(f"Unknown integrator: {integrator}")

        # Update x0 to concatenated initial states
        x0_parts = [self._submodels[name].x0.as_vec() for name in self._submodels.keys()]
        self.x0_composed = ca.vertcat(*x0_parts) if x0_parts else ca.DM.zeros(0, 1)

        # Create structured initial state with submodel access
        composite_fields = []
        for name, submodel in self._submodels.items():
            composite_fields.append(
                (
                    name,
                    type(submodel.x0),
                    dc_field(default_factory=lambda sm=submodel: copy.deepcopy(sm.x0)),
                )
            )

        def _as_vec(self_state):
            return ca.vertcat(*[getattr(self_state, name).as_vec() for name in self._submodels.keys()])

        def _from_vec(cls_obj, x_vec):
            kwargs = {}
            for name, (start, end) in self._submodel_state_slices.items():
                submodel = self._submodels[name]
                x_sub_vec = x_vec[start:end]
                kwargs[name] = type(submodel.x0).from_vec(x_sub_vec)
            return cls_obj(**kwargs)

        ComposedState = make_dataclass(
            "ComposedState",
            composite_fields,
            namespace={"as_vec": _as_vec, "from_vec": classmethod(_from_vec)},
        )

        self.x0 = ComposedState()
        self._state_to_vec = lambda x_struct: x_struct.as_vec()
        self._vec_to_state = lambda x_vec: ComposedState.from_vec(x_vec)

    def _extract_field_from_vec(self, vec, struct_obj, field_name):
        """Extract a specific field from a CasADi vector given the structure."""
        offset = 0
        for fld in fields(struct_obj):
            if fld.name == field_name:
                fld_val = getattr(struct_obj, fld.name)
                fld_size = fld_val.shape[0] if hasattr(fld_val, "shape") else 1
                return vec[offset : offset + fld_size]
            else:
                fld_val = getattr(struct_obj, fld.name)
                fld_size = fld_val.shape[0] if hasattr(fld_val, "shape") else 1
                offset += fld_size
        return ca.DM.zeros(1, 1)  # Fallback

    def _build_composed_outputs(self, submodel_state_slices, submodel_outputs):
        """Build output function for composed model.

        Creates f_y that evaluates all submodel outputs and routes them to parent outputs
        according to the output connections defined.
        """
        # Create symbolic variables for function signature
        total_states = sum(end - start for start, end in submodel_state_slices.values())
        x_combined = ca.SX.sym("x_combined", total_states)
        u_parent = self.u.as_vec()
        p_parent = self.p.as_vec()

        # Extract submodel states from combined vector
        submodel_states = {}
        for name, (start, end) in submodel_state_slices.items():
            submodel_states[name] = x_combined[start:end]

        # Build input vectors for each submodel (same logic as in build_composed)
        submodel_inputs = {}

        # First pass: Build preliminary inputs with state and parent input connections
        for name, submodel in self._submodels.items():
            u_sub_dict = {}
            input_conns = self._input_connections.get(name, {})

            for field_obj in fields(submodel.u):
                field_name = field_obj.name
                full_path = f"{name}.u.{field_name}"

                if full_path in input_conns:
                    source = input_conns[full_path]

                    if source.startswith("u."):
                        # Parent input connection
                        parent_field = source[2:]
                        if hasattr(self.u, parent_field):
                            u_sub_dict[field_name] = getattr(self.u, parent_field)
                        else:
                            u_sub_dict[field_name] = getattr(submodel.u0, field_name)
                    else:
                        parts = source.split(".", 2)
                        if len(parts) >= 3 and parts[1] == "x":
                            # State connection
                            source_model, source_type, source_field = parts
                            if source_model in submodel_states:
                                u_sub_dict[field_name] = self._extract_field_from_vec(
                                    submodel_states[source_model],
                                    self._submodels[source_model].x,
                                    source_field,
                                )
                            else:
                                u_sub_dict[field_name] = getattr(submodel.u0, field_name)
                        else:
                            # Output connection - can't resolve yet, use default
                            u_sub_dict[field_name] = getattr(submodel.u0, field_name)
                else:
                    u_sub_dict[field_name] = getattr(submodel.u0, field_name)

            u_sub_list = [u_sub_dict[f.name] for f in fields(submodel.u)]
            submodel_inputs[name] = ca.vertcat(*u_sub_list) if u_sub_list else ca.DM.zeros(0, 1)

        # Evaluate all submodel outputs
        evaluated_outputs = {}
        for name, submodel in self._submodels.items():
            if hasattr(submodel, "f_y"):
                x_sub = submodel_states[name]
                u_sub = submodel_inputs[name]
                p_sub = submodel.p0.as_vec()
                y_sub = submodel.f_y(x_sub, u_sub, p_sub)
                evaluated_outputs[name] = y_sub

        # Second pass: Resolve output connections and rebuild inputs
        for name, submodel in self._submodels.items():
            u_sub_dict = {}
            input_conns = self._input_connections.get(name, {})

            for field_obj in fields(submodel.u):
                field_name = field_obj.name
                full_path = f"{name}.u.{field_name}"

                if full_path in input_conns:
                    source = input_conns[full_path]

                    if source.startswith("u."):
                        # Parent input
                        parent_field = source[2:]
                        if hasattr(self.u, parent_field):
                            u_sub_dict[field_name] = getattr(self.u, parent_field)
                        else:
                            u_sub_dict[field_name] = getattr(submodel.u0, field_name)

                    elif "." in source:
                        parts = source.split(".", 2)
                        if len(parts) >= 3:
                            source_model, source_type, source_field = parts

                            if source_type == "y" and source_model in evaluated_outputs:
                                u_sub_dict[field_name] = self._extract_field_from_vec(
                                    evaluated_outputs[source_model],
                                    self._submodels[source_model].y,
                                    source_field,
                                )
                            elif source_type == "x" and source_model in submodel_states:
                                u_sub_dict[field_name] = self._extract_field_from_vec(
                                    submodel_states[source_model],
                                    self._submodels[source_model].x,
                                    source_field,
                                )
                            else:
                                u_sub_dict[field_name] = getattr(submodel.u0, field_name)
                        else:
                            u_sub_dict[field_name] = getattr(submodel.u0, field_name)
                    else:
                        u_sub_dict[field_name] = getattr(submodel.u0, field_name)
                else:
                    u_sub_dict[field_name] = getattr(submodel.u0, field_name)

            u_sub_list = [u_sub_dict[f.name] for f in fields(submodel.u)]
            submodel_inputs[name] = ca.vertcat(*u_sub_list) if u_sub_list else ca.DM.zeros(0, 1)

        # Re-evaluate outputs with updated inputs (for output-to-input dependencies)
        for name, submodel in self._submodels.items():
            if hasattr(submodel, "f_y"):
                x_sub = submodel_states[name]
                u_sub = submodel_inputs[name]
                p_sub = submodel.p0.as_vec()
                y_sub = submodel.f_y(x_sub, u_sub, p_sub)
                evaluated_outputs[name] = y_sub

        # Build parent output vector by routing from submodel outputs
        y_parent_dict = {}
        for field_obj in fields(self.y):
            field_name = field_obj.name
            # Find the source for this output field
            source_found = False

            for submodel_name, output_conns in self._output_connections.items():
                for source_path, target_path in output_conns.items():
                    if target_path == f"y.{field_name}":
                        # Parse source path: "submodel.y.field"
                        parts = source_path.split(".", 2)
                        if len(parts) == 3 and parts[0] in evaluated_outputs:
                            source_model, source_type, source_field = parts
                            y_parent_dict[field_name] = self._extract_field_from_vec(
                                evaluated_outputs[source_model],
                                self._submodels[source_model].y,
                                source_field,
                            )
                            source_found = True
                            break
                if source_found:
                    break

            if not source_found:
                # Use default value if no connection found
                y_parent_dict[field_name] = getattr(self.y, field_name)

        # Build output vector in correct field order
        y_parent_list = [y_parent_dict[f.name] for f in fields(self.y)]
        y_parent_combined = ca.vertcat(*y_parent_list) if y_parent_list else ca.DM.zeros(0, 1)

        # Create the composed output function
        self.f_y = ca.Function(
            "f_y_composed",
            [x_combined, u_parent, p_parent],
            [y_parent_combined],
            ["x", "u", "p"],
            ["y"],
        )
