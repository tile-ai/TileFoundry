# TileFoundry Spec - Parser

The Parser accepts authored Python functions and produces HIR or TIR through one typed API.

## 1. Public API

`@module` executes its Python class body and finalizes the collected Function declarations,
child Modules, and ordinary methods. Module authoring is two-phase: class execution records
Function, specialization, and converter declarations; finalization attaches all child Modules
and puts them in parser scope before parsing Functions in source order. `@func` produces an HIR
Function; `@prim_func` produces a TIR PrimFunction.
`specialize` and `converter` register variants and weight converters on an existing HIR Function.

```python
def parse_function(
    fn: FunctionType, context: FuncParserContext
) -> hir.Function | tir.PrimFunction: ...
```

`FuncParserContext` carries the dialect, Function role, closure, topology scope, target, and
optional base/key for one parse. `FunctionRole` is `ROOT`, `VARIANT`, or `CONVERTER`.
`ParseError` is the single authored-source diagnostic type and includes source location and
recursive parse situation. These are the only public parser symbols.

## 2. Syntax and Rules

### 2.1 Syntax

<!-- parser-grammar:start -->
```ebnf
; root: function
; literal: Python ast.Constant syntax, e.g. 1, "bf16", or None
; name: Python variable name; primary: name/attribute base for calls and subscripts
; expression: Python syntax composed from literals, names, primaries, and operators
; runtime-expression: expression lowered to a TileFoundry IR Expr
mesh-axis             ::= identifier
                          | identifier '.' identifier
dim-expr              ::= integer-literal
                          | identifier
                          | primary '.' identifier
                          | dim-expr ('+' | '-' | '*' | '//' | '%') dim-expr
                          | (identifier | primary '.' identifier) '(' (dim-expr (',' dim-expr)*)?
                            ')'
placed-layout         ::= '(' ((expression '@' ('(' mesh-axis (',' mesh-axis)* ')' | mesh-axis) |
                          dim-expr) (',' (expression '@' ('(' mesh-axis (',' mesh-axis)* ')' |
                          mesh-axis) | dim-expr))*)? ')'
shape                 ::= '(' (dim-expr (',' dim-expr)*)? ')'
                          | identifier
                          | primary '.' identifier
tensor-shape-layout   ::= placed-layout
                          | shape
dtype                 ::= string-literal
                          | primary
literal               ::= None
                          | Ellipsis
                          | boolean-literal
                          | integer-literal
                          | float-literal
                          | complex-literal
                          | string-literal
                          | bytes-literal
primary               ::= identifier
                          | primary '.' identifier
sequence              ::= '(' (expression (',' expression)*)? ')'
                          | '[' (expression (',' expression)*)? ']'
                          | '{' (expression (',' expression)*)? '}'
dict                  ::= '{' (expression ':' expression (',' expression ':' expression)*)? '}'
binary-operation      ::= expression ('+' | '-' | '*' | '/' | '//' | '%' | '**') expression
unary-operation       ::= ('+' | '-' | 'not') expression
slice                 ::= (expression)? ':' (expression)? (':' expression)?
subscript             ::= expression '[' expression ']'
expression            ::= literal
                          | primary
                          | sequence
                          | dict
                          | binary-operation
                          | unary-operation
                          | call
                          | slice
                          | subscript
call                  ::= expression '(' ((expression | keyword-name '=' expression) (','
                          (expression | keyword-name '=' expression))*)? ')'
explicit-layout       ::= '(' (tensor-shape-layout | shape) ',' shape ')'
plain-layout          ::= '(' (dim-expr (',' dim-expr)*)? ')'
layout                ::= None
                          | primary
                          | call
                          | explicit-layout
                          | placed-layout
                          | plain-layout
storage               ::= string-literal
                          | primary
tensor-optional-slot  ::= layout
                          | storage
tensor                ::= tensor-head '[' '(' (tensor-shape-layout ',' dtype | tensor-shape-layout
                          ',' dtype ',' tensor-optional-slot | tensor-shape-layout ',' dtype ','
                          tensor-optional-slot ',' tensor-optional-slot) ')' ']'
scalar-type           ::= primary
type-annotation       ::= tensor
                          | scalar-type
signature             ::= (name ':' type-annotation (',' name ':' type-annotation)*)?
return-type           ::= type-annotation
loop-carry-statement  ::= expression '=' expression
                          | 'for' name 'in' expression ':' loop-carry
                          | statement
loop-carry            ::= (loop-carry-statement (newline loop-carry-statement)*)?
loop-header           ::= 'for' identifier 'in' ('tile' | 'range') '(' expression (',' expression)*
                          ')' ':' loop-carry
loop-body             ::= (statement (newline statement)*)?
for                   ::= 'for' name 'in' expression ':' loop-body
mesh-context          ::= ('Mesh' | primary '.' identifier) '(' (expression | ('layout' | 'names')
                            '=' expression) (',' (expression | ('layout' | 'names') '='
                            expression))* ')'
                          | expression
with                  ::= 'with' mesh-context ('as' identifier)? ':' block
op-call               ::= primary '(' ((expression | keyword-name '=' expression) (',' (expression |
                          keyword-name '=' expression))*)? ')'
launch                ::= callee '(' ')'
slice-endpoint-binary ::= index-endpoint dim-op index-endpoint
mesh-coordinate       ::= identifier '.' identifier
index-endpoint        ::= literal
                          | primary
                          | slice-endpoint-binary
                          | mesh-coordinate
                          | runtime-expression
                          | expression
index-slice           ::= (index-endpoint)? ':' (index-endpoint)? (':' index-endpoint)?
subscript-index       ::= '(' ((index-slice | index-endpoint) (',' (index-slice |
                            index-endpoint))*)? ')'
                          | index-slice
                          | index-endpoint
subscript-expression  ::= runtime-expression '[' subscript-index ']'
binary-expression     ::= runtime-expression binary-op runtime-expression
                          | runtime-expression comparison-op runtime-expression
                          | runtime-expression boolean-op runtime-expression
unary-expression      ::= unary-op runtime-expression
name                  ::= identifier
constant              ::= boolean-literal
                          | integer-literal
                          | float-literal
tuple-expression      ::= '(' (runtime-expression (',' runtime-expression)*)? ')'
runtime-expression    ::= op-call
                          | launch
                          | subscript-expression
                          | binary-expression
                          | unary-expression
                          | mesh-coordinate
                          | name
                          | constant
                          | tuple-expression
                          | tensor
                          | primary '.' identifier
tuple-assignment      ::= '(' identifier (',' identifier)* ')' '=' runtime-expression
where-annotation      ::= 'where' '(' ')'
statement             ::= for
                          | with
                          | tuple-assignment
                          | identifier '=' (runtime-expression | expression)
                          | identifier ':' (where-annotation | type-annotation) ('='
                            (runtime-expression | expression))?
                          | 'return' (runtime-expression)?
                          | runtime-expression
                          | 'pass'
block                 ::= (statement (newline statement)*)?
function              ::= 'def' name '(' signature ')' ('->' return-type)? ':' block
```
<!-- parser-grammar:end -->

### 2.2 Rules

<!-- parser-constraints:start -->
| Owner | Situation | Rule | Statement | Source |
| --- | --- | --- | --- | --- |
| binary_expression | expression | CallBindingRule | A call must bind its arguments into a Call tuple. | src/tilefoundry/parser/pattern_nodes.py |
| binary_expression | expression | CallExpectedTypeRule | A call's inferred type must satisfy the expected expression type. | src/tilefoundry/parser/pattern_nodes.py |
| binary_expression | expression | CallTypeInferenceRule | A call's result type must be inferred from its binding. | src/tilefoundry/parser/pattern_nodes.py |
| binary_expression | slice_endpoint | CallBindingRule | A call must bind its arguments into a Call tuple. | src/tilefoundry/parser/pattern_nodes.py |
| binary_expression | slice_endpoint | CallExpectedTypeRule | A call's inferred type must satisfy the expected expression type. | src/tilefoundry/parser/pattern_nodes.py |
| binary_expression | slice_endpoint | CallTypeInferenceRule | A call's result type must be inferred from its binding. | src/tilefoundry/parser/pattern_nodes.py |
| binary_expression | subscript_index | CallBindingRule | A call must bind its arguments into a Call tuple. | src/tilefoundry/parser/pattern_nodes.py |
| binary_expression | subscript_index | CallExpectedTypeRule | A call's inferred type must satisfy the expected expression type. | src/tilefoundry/parser/pattern_nodes.py |
| binary_expression | subscript_index | CallTypeInferenceRule | A call's result type must be inferred from its binding. | src/tilefoundry/parser/pattern_nodes.py |
| dim_expr | dim_expr | ShapeDimRule | A shape dimension must be an integer, DimVar, or expression. | src/tilefoundry/parser/ast_pattern.py |
| dim_expr | layout_extent | ShapeDimRule | A shape dimension must be an integer, DimVar, or expression. | src/tilefoundry/parser/ast_pattern.py |
| dim_expr | layout_shape | ShapeDimRule | A shape dimension must be an integer, DimVar, or expression. | src/tilefoundry/parser/ast_pattern.py |
| dim_expr | tensor_dim_expr | ShapeDimRule | A shape dimension must be an integer, DimVar, or expression. | src/tilefoundry/parser/ast_pattern.py |
| dim_expr | tensor_optional_slot | ShapeDimRule | A shape dimension must be an integer, DimVar, or expression. | src/tilefoundry/parser/ast_pattern.py |
| dim_expr | tensor_shape | ShapeDimRule | A shape dimension must be an integer, DimVar, or expression. | src/tilefoundry/parser/ast_pattern.py |
| dtype | tensor_dtype | CanonicalDTypeRule | A dtype must resolve to a canonical DType. | src/tilefoundry/parser/ast_pattern.py |
| explicit_layout | tensor_optional_slot | LayoutPositionRule | A layout must be legal for its parser position. | src/tilefoundry/parser/ast_pattern.py |
| explicit_layout | tensor_optional_slot | LayoutShapeRule | A layout must have a valid non-boolean shape. | src/tilefoundry/parser/ast_pattern.py |
| function | function | FunctionDialectRule | A function kind and constructed value must agree with the active dialect. | src/tilefoundry/parser/pattern_nodes.py |
| function | function | FunctionRegistrationRule | A validated function must be registered exactly once in its owning scope. | src/tilefoundry/parser/pattern_nodes.py |
| function | function | FunctionReturnRule | A HIR function body's inferred type must match its return type. | src/tilefoundry/parser/pattern_nodes.py |
| function | function | FunctionRoleValidationRule | A root, variant, or converter must satisfy its role before registration. | src/tilefoundry/parser/pattern_nodes.py |
| function | function | FunctionSignatureRule | A function must construct an ordered parameter tuple. | src/tilefoundry/parser/pattern_nodes.py |
| layout | tensor_optional_slot | LayoutPositionRule | A layout must be legal for its parser position. | src/tilefoundry/parser/ast_pattern.py |
| layout | tensor_optional_slot | LayoutShapeRule | A layout must have a valid non-boolean shape. | src/tilefoundry/parser/ast_pattern.py |
| op_call | expression | CallBindingRule | A call must bind its arguments into a Call tuple. | src/tilefoundry/parser/pattern_nodes.py |
| op_call | expression | CallExpectedTypeRule | A call's inferred type must satisfy the expected expression type. | src/tilefoundry/parser/pattern_nodes.py |
| op_call | expression | CallTypeInferenceRule | A call's result type must be inferred from its binding. | src/tilefoundry/parser/pattern_nodes.py |
| op_call | slice_endpoint | CallBindingRule | A call must bind its arguments into a Call tuple. | src/tilefoundry/parser/pattern_nodes.py |
| op_call | slice_endpoint | CallExpectedTypeRule | A call's inferred type must satisfy the expected expression type. | src/tilefoundry/parser/pattern_nodes.py |
| op_call | slice_endpoint | CallTypeInferenceRule | A call's result type must be inferred from its binding. | src/tilefoundry/parser/pattern_nodes.py |
| op_call | subscript_index | CallBindingRule | A call must bind its arguments into a Call tuple. | src/tilefoundry/parser/pattern_nodes.py |
| op_call | subscript_index | CallExpectedTypeRule | A call's inferred type must satisfy the expected expression type. | src/tilefoundry/parser/pattern_nodes.py |
| op_call | subscript_index | CallTypeInferenceRule | A call's result type must be inferred from its binding. | src/tilefoundry/parser/pattern_nodes.py |
| placed_layout | layout_shape | LayoutPositionRule | A layout must be legal for its parser position. | src/tilefoundry/parser/ast_pattern.py |
| placed_layout | layout_shape | LayoutShapeRule | A layout must have a valid non-boolean shape. | src/tilefoundry/parser/ast_pattern.py |
| placed_layout | tensor_optional_slot | LayoutPositionRule | A layout must be legal for its parser position. | src/tilefoundry/parser/ast_pattern.py |
| placed_layout | tensor_optional_slot | LayoutShapeRule | A layout must have a valid non-boolean shape. | src/tilefoundry/parser/ast_pattern.py |
| placed_layout | tensor_shape | LayoutPositionRule | A layout must be legal for its parser position. | src/tilefoundry/parser/ast_pattern.py |
| placed_layout | tensor_shape | LayoutShapeRule | A layout must have a valid non-boolean shape. | src/tilefoundry/parser/ast_pattern.py |
| plain_layout | tensor_optional_slot | LayoutPositionRule | A layout must be legal for its parser position. | src/tilefoundry/parser/ast_pattern.py |
| plain_layout | tensor_optional_slot | LayoutShapeRule | A layout must have a valid non-boolean shape. | src/tilefoundry/parser/ast_pattern.py |
| shape | layout_shape | ShapeTupleRule | A shape must construct a tuple of dimensions. | src/tilefoundry/parser/ast_pattern.py |
| shape | layout_strides | ShapeTupleRule | A shape must construct a tuple of dimensions. | src/tilefoundry/parser/ast_pattern.py |
| shape | tensor_shape | ShapeTupleRule | A shape must construct a tuple of dimensions. | src/tilefoundry/parser/ast_pattern.py |
| storage | tensor_optional_slot | StorageValueRule | Storage must resolve to a StorageKind. | src/tilefoundry/parser/ast_pattern.py |
| tensor | annotation | TensorLayoutStorageRule | A tensor type must contain compatible layout and storage values. | src/tilefoundry/parser/ast_pattern.py |
| tensor | annotation | TensorPositionRule | A tensor type's storage must be legal for its dialect and position. | src/tilefoundry/parser/ast_pattern.py |
| tensor | expression | TensorLayoutStorageRule | A tensor type must contain compatible layout and storage values. | src/tilefoundry/parser/ast_pattern.py |
| tensor | expression | TensorPositionRule | A tensor type's storage must be legal for its dialect and position. | src/tilefoundry/parser/ast_pattern.py |
| tensor | slice_endpoint | TensorLayoutStorageRule | A tensor type must contain compatible layout and storage values. | src/tilefoundry/parser/ast_pattern.py |
| tensor | slice_endpoint | TensorPositionRule | A tensor type's storage must be legal for its dialect and position. | src/tilefoundry/parser/ast_pattern.py |
| tensor | subscript_index | TensorLayoutStorageRule | A tensor type must contain compatible layout and storage values. | src/tilefoundry/parser/ast_pattern.py |
| tensor | subscript_index | TensorPositionRule | A tensor type's storage must be legal for its dialect and position. | src/tilefoundry/parser/ast_pattern.py |
| tensor | type_annotation | TensorLayoutStorageRule | A tensor type must contain compatible layout and storage values. | src/tilefoundry/parser/ast_pattern.py |
| tensor | type_annotation | TensorPositionRule | A tensor type's storage must be legal for its dialect and position. | src/tilefoundry/parser/ast_pattern.py |
| unary_expression | expression | CallBindingRule | A call must bind its arguments into a Call tuple. | src/tilefoundry/parser/pattern_nodes.py |
| unary_expression | expression | CallExpectedTypeRule | A call's inferred type must satisfy the expected expression type. | src/tilefoundry/parser/pattern_nodes.py |
| unary_expression | expression | CallTypeInferenceRule | A call's result type must be inferred from its binding. | src/tilefoundry/parser/pattern_nodes.py |
| unary_expression | slice_endpoint | CallBindingRule | A call must bind its arguments into a Call tuple. | src/tilefoundry/parser/pattern_nodes.py |
| unary_expression | slice_endpoint | CallExpectedTypeRule | A call's inferred type must satisfy the expected expression type. | src/tilefoundry/parser/pattern_nodes.py |
| unary_expression | slice_endpoint | CallTypeInferenceRule | A call's result type must be inferred from its binding. | src/tilefoundry/parser/pattern_nodes.py |
| unary_expression | subscript_index | CallBindingRule | A call must bind its arguments into a Call tuple. | src/tilefoundry/parser/pattern_nodes.py |
| unary_expression | subscript_index | CallExpectedTypeRule | A call's inferred type must satisfy the expected expression type. | src/tilefoundry/parser/pattern_nodes.py |
| unary_expression | subscript_index | CallTypeInferenceRule | A call's result type must be inferred from its binding. | src/tilefoundry/parser/pattern_nodes.py |
| module | module_function | ModuleFunctionValidationRule | A module function must satisfy its root, variant, or converter role before mutation. | src/tilefoundry/parser/ast_pattern.py |
| module | module_function | ModuleFunctionRegistrationRule | A validated module function must be recorded in declaration order. | src/tilefoundry/parser/ast_pattern.py |
| module | module_finalization | ModuleFinalizationRule | A module declaration must contain valid unique members and a resolvable entry. | src/tilefoundry/parser/ast_pattern.py |

<!-- parser-constraints:end -->

## 3. Implementation Overview

| Component | Responsibility |
| --- | --- |
| Parser API and Context | Receives authored Functions and carries dialect, role, scope, and recursion inputs. |
| Executable Pattern Graph | Composes concrete AST elements into the Function root pattern. |
| Match and Construction | Matches recursively into `AstMatch`, then constructs owner values on return. |
| Ordered Rules | Validates and normalizes each owner value after construction. |
| Module Build | Lets Python execute the class body, collects declarations, resolves child Modules first, then parses Functions in source order and finalizes the Module. |
| Pattern Visitor | Traverses the same graph to render this section's generated grammar and constraints. |

```mermaid
classDiagram
    ParserAPI --> FuncParserContext
    ParserAPI --> FunctionPattern
    AstPattern <|.. Element
    Element o-- AstPattern
    Element o-- AstRule
    AstPattern --> AstMatch
    PatternVisitor ..> AstPattern
    ParserAPI ..> ModuleBuild
```

```mermaid
flowchart TD
    API["parse_function(fn, context)"] --> AST["Extract FunctionDef AST"]
    AST --> ROOT["FunctionPattern.match"]
    ROOT --> TREE["AstMatch tree"]
    TREE --> BACKWARD["construct children, then apply Rules"]
    BACKWARD --> FUNCTION["HIR Function / TIR PrimFunction"]
    FUNCTION --> MODULE{"Module authoring context?"}
    MODULE -->|yes| FINALIZE["defer declaration"]
    FINALIZE --> CHILDREN["attach child Modules and bind module scope"]
    CHILDREN --> ORDERED["parse roots in source order; then variants/converters"]
    ORDERED --> BUILT["construct final Module and verify"]
    BUILT --> RETURN
    MODULE -->|no| RETURN["return standalone result"]
```

Pattern combinators serve both runtime matching and Spec traversal. `AstMatch` separates syntax
matching from object construction, while each Rule reads the recursive context after its owner
value exists. Module class control flow remains Python execution; no Module AST grammar exists.
