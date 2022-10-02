use pyo3::prelude::*;
extern crate wake;
use polars::prelude::DataFrame;
use wake::graph::*;

pub mod tpch;
pub mod utils;


use tpch::*;

use arrow::ffi;
use std::time;
use std::thread;
use polars::prelude::*;
use polars_arrow::export::arrow;
use pyo3::exceptions::PyValueError;
use pyo3::ffi::Py_uintptr_t;
use pyo3::{PyAny, PyObject, PyResult};
use pyo3::types::{PyDict, PyString};

/// Take an arrow array from python and convert it to a rust arrow array.
/// This operation does not copy data.
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

pub fn rust_series_to_py_series(series: &Series) -> PyResult<PyObject> {
    // ensure we have a single chunk
    let series = series.rechunk();
    let array = series.to_arrow(0);

    // acquire the gil
    #[allow(deprecated)]
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
fn main() {
    // Arguments:
    // 0: Whether to run query or test. Required. query/test.
    // 1: Query Number. Required.
    // 2: Scale of the TPC-H Dataset. Optional. Default: 1.
    // 3: Directory containing the dataset. Optional. Default: resources/tpc-h/data/scale=1/partition=1/

    env_logger::Builder::from_default_env()
        .format_timestamp_micros()
        .init();

    let q1 = String::from("q1");

    let data_directory = "../../resources/tpc-h/data/scale=1/partition=1";
    let mut output_reader = NodeReader::empty();
    let mut query_service = get_query_service(&q1, 1, data_directory, &mut output_reader);
    log::info!("Running Query: {}", q1);

    query_service.run();
    // YOU WANT TO INITIALIZE THE FIGURE HERE, OUTSIDE THE LOOP.
    #[allow(deprecated)]
    let gil = Python::acquire_gil();
    let py = gil.python();
    let locals = PyDict::new(py);
    py.run(r#"
fig = plt.figure(figsize=(5, 4), layout='tight', clip_on=True)
ax = fig.add_subplot(111)
fig.show()
"#, None, Some(locals)).unwrap();

    let py_fig = locals.get_item_with_error("fig").unwrap();
    let py_ax = locals.get_item_with_error("ax").unwrap();

    loop {
        // let res = py.allow_threads(move || {
            let message = output_reader.read();
            if message.is_eof() {
                // return Err(message);
                break;
            }
            // Ok(message)
        // });
        // let message = match res {
        //     Ok(message) => message,
        //     Err(error) => break,
        // };
        let df = message.datablock().data();
        #[allow(deprecated)]
        let gil = Python::acquire_gil();
        let py = gil.python();
        py.run("print(\"HERE\")", None, None).unwrap();

        let dict = PyDict::new(py);
        let mut all_py_series = Vec::new();
        for col in df.get_columns() {
            let py_ser = rust_series_to_py_series(col).unwrap();
            let py_name = PyString::new(py, col.name());
            dict.set_item(py_name, &py_ser).unwrap();
            all_py_series.push(py_ser);
        }
        let pandas = py.import("pandas").unwrap();
        let py_df = pandas.call_method1("DataFrame", (dict,)).unwrap();

        let locals = PyDict::new(py);
        locals.set_item("df", py_df).unwrap();
        locals.set_item("ax", py_ax).unwrap();
        locals.set_item("fig", py_fig).unwrap();
        py.run(r#"

df.loc[0, 'sum_qty'] = random.randint(2974251, 223428120)
ax.clear()
df.plot(x='count_order', y='sum_qty', ax=ax, kind='bar')
fig.canvas.draw()
"#, None, Some(locals)).unwrap();
        
        let two_sec = time::Duration::from_millis(2000);
        thread::sleep(two_sec);

        // let plt = py.import("matplotlib.pyplot").unwrap();
        // py_df.call_method("bar");
        // plt.call_method1("bar", (&all_py_series[0], &all_py_series[1],)).unwrap();
        // query_result.push(df.clone());
    }
    query_service.join();
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
    m.add_function(wrap_pyfunction!(main, m)?)?;
    Ok(())
}
