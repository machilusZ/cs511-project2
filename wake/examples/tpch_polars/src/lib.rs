use pyo3::prelude::*;
extern crate wake;
use polars::prelude::*;
use polars::frame::DataFrame;
use std::env;
use wake::graph::*;

pub mod tpch;
pub mod utils;

use tpch::*;

use polars_arrow::export::arrow;
use pyo3::exceptions::PyValueError;
use pyo3::ffi::Py_uintptr_t;
use pyo3::prelude::*;
use pyo3::{PyAny, PyObject, PyResult};
use pyo3::types::{PyDict, PyString};
use arrow::ffi;

pub fn py_series_to_rust_series(series: &PyAny) -> PyResult<Series> {
    // rechunk series so that they have a single arrow array
    let series = series.call_method0("rechunk")?;

    let name = series.getattr("name")?.extract::<String>()?;

    // retrieve pyarrow array
    let array = series.call_method0("to_arrow")?;

    // retrieve rust arrow array
    let array = array_to_rust(array)?;

    Series::try_from((name.as_str(), array)).map_err(|e| PyValueError::new_err(format!("{}", e)))
}

/// Arrow array to Python.
pub(crate) fn to_py_array(py: Python, pyarrow: &PyModule, array: ArrayRef) -> PyResult<PyObject> {
    let schema = Box::new(ffi::export_field_to_c(&ArrowField::new(
        "",
        array.data_type().clone(),
        true,
    )));
    let array = Box::new(ffi::export_array_to_c(array));

    let schema_ptr: *const ffi::ArrowSchema = &*schema;
    let array_ptr: *const ffi::ArrowArray = &*array;

    let array = pyarrow.getattr("Array")?.call_method1(
        "_import_from_c",
        (array_ptr as Py_uintptr_t, schema_ptr as Py_uintptr_t),
    )?;

    Ok(array.to_object(py))
}

fn array_to_rust(arrow_array: &PyAny) -> PyResult<ArrayRef> {
    // prepare a pointer to receive the Array struct
    let array = Box::new(ffi::ArrowArray::empty());
    let schema = Box::new(ffi::ArrowSchema::empty());

    let array_ptr = &*array as *const ffi::ArrowArray;
    let schema_ptr = &*schema as *const ffi::ArrowSchema;

    // make the conversion through PyArrow's private API
    // this changes the pointer's memory and is thus unsafe. In particular, `_export_to_c` can go out of bounds
    arrow_array.call_method1(
        "_export_to_c",
        (array_ptr as Py_uintptr_t, schema_ptr as Py_uintptr_t),
    )?;

    unsafe {
        let field = ffi::import_field_from_c(schema.as_ref()).unwrap();
        let array = ffi::import_array_from_c(*array, field.data_type).unwrap();
        Ok(array.into())
    }
}

pub fn rust_series_to_py_series(series: &Series) -> PyResult<PyObject> {
    // ensure we have a single chunk
    let series = series.rechunk();
    let array = series.to_arrow(0);

    // acquire the gil
    let gil = Python::acquire_gil();
    let py = gil.python();
    // import pyarrow
    let pyarrow = py.import("pyarrow")?;

    // pyarrow array
    let pyarrow_array = to_py_array(py, pyarrow, array)?;

    // import polars
    let polars = py.import("polars")?;
    let out = polars.call_method1("from_arrow", (pyarrow_array,))?;
    Ok(out.to_object(py))
}

#[pyfunction]
fn test_function() -> String {
    "Hello World from Rust".to_string()
}

#[pyfunction]
fn main() -> PyResult<PyObject> {
    // Arguments:
    // 0: Whether to run query or test. Required. query/test.
    // 1: Query Number. Required.
    // 2: Scale of the TPC-H Dataset. Optional. Default: 1.
    // 3: Directory containing the dataset. Optional. Default: resources/tpc-h/data/scale=1/partition=1/

    env_logger::Builder::from_default_env()
        .format_timestamp_micros()
        .init();

    let args = env::args().skip(1).collect::<Vec<String>>();
    let mut my_vec: Vec<String> = Vec::new();
    let mut my_vec1: Vec<String> = Vec::new();
    let name1 = String::from("query");
    let name2 = String::from("q1");
    my_vec.push(name1);
    my_vec1.push(name2);
    run_query(my_vec1)
}


fn run_query(args: Vec<String>) -> PyResult<PyObject> {
    if args.len() == 0 {
        panic!("Query not specified. Run like: cargo run --release --example tpch_polars -- q1")
    }
    let query_no = args[0].as_str();
    let scale = if args.len() <= 1 {
        1
    } else {
        *(&args[1].parse::<usize>().unwrap())
    };
    let data_directory = if args.len() <= 2 {
        "../../resources/tpc-h/data/scale=1/partition=1"
    } else {
        args[2].as_str()
    };
    let mut output_reader = NodeReader::empty();
    let mut query_service = get_query_service(query_no, scale, data_directory, &mut output_reader);
    log::info!("Running Query: {}", query_no);
    let dfs: Vec<DataFrame> = utils::run_query(&mut query_service, &mut output_reader);
    let df = &dfs[0];
    let ser1 = &df[0];
    let ser2 = &df[1];
    let py_ser1 = rust_series_to_py_series(ser1)?;
    let py_ser2 = rust_series_to_py_series(ser2)?;

    let gil = Python::acquire_gil();
    let py = gil.python();
    let dict = PyDict::new(py);
    let name1 = PyString::new(py, ser1.name());
    let name2 = PyString::new(py, ser2.name());
    dict.set_item(name1, py_ser1);
    dict.set_item(name2, py_ser2);
    let pandas = py.import("pandas")?;
    let out = pandas.call_method1("DataFrame", (dict,))?;
    Ok(out.to_object(py))
}

pub fn get_query_service(
    query_no: &str,
    scale: usize,
    data_directory: &str,
    output_reader: &mut NodeReader<DataFrame>,
) -> ExecutionService<DataFrame> {
    let table_input = utils::load_tables(data_directory, scale);
    // TODO: UNCOMMENT THE MATCH STATEMENTS BELOW AS YOU IMPLEMENT THESE QUERIES.
    let query_service = match query_no {
        "q1" => q1::query(table_input, output_reader),
        "q14" => q14::query(table_input, output_reader),
        _ => panic!("Invalid Query Parameter"),
    };
    query_service
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn tpch_polars(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(test_function, m)?)?;
    m.add_function(wrap_pyfunction!(main, m)?)?;
    Ok(())
}