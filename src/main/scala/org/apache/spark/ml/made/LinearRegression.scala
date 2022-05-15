package org.apache.spark.ml.made

import breeze.linalg.{DenseVector => BDenseVector}
import breeze.optimize.LBFGSB
import org.apache.spark.ml.PredictorParams
import org.apache.spark.ml.linalg.{DenseVector, Vector, VectorUDT, Vectors}
import org.apache.spark.ml.optim.aggregator.LeastSquaresAggregator
import org.apache.spark.ml.optim.loss.{DifferentiableRegularization, L2Regularization, RDDLossFunction}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.shared.{HasMaxIter, HasRegParam, HasTol, HasWeightCol}
import org.apache.spark.ml.regression.{RegressionModel, Regressor}
import org.apache.spark.ml.util._
import org.apache.spark.mllib
import org.apache.spark.mllib.stat.MultivariateOnlineSummarizer
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.types.{DoubleType, StructType}

trait LinearRegressionParams extends PredictorParams with HasRegParam with HasMaxIter with HasTol with HasWeightCol {
  def setOutputCol(value: String): this.type = set(labelCol, value)
  def setMaxIterCol(value: Int): this.type = set(maxIter, value)
  def setTolCol(value: Double): this.type = set(tol, value)
  def setRegCol(value: Double): this.type = set(regParam, value)

  protected def validateAndTransformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, getFeaturesCol, new VectorUDT())

    var newSchema = SchemaUtils.appendColumn(schema, $(predictionCol), DoubleType)

    if (newSchema.fieldNames.contains($(labelCol)))
      SchemaUtils.checkColumnType(newSchema, getLabelCol, DoubleType)
    else
      newSchema = SchemaUtils.appendColumn(newSchema, $(labelCol), DoubleType)

    newSchema
  }
}

class LinearRegression(override val uid: String) extends Regressor[Vector, LinearRegression, LinearRegressionModel]
  with LinearRegressionParams
  with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("linearRegression"))

  override def train(dataset: Dataset[_]): LinearRegressionModel = {
    val numFeatures = MetadataUtils.getNumFeatures(dataset, $(featuresCol))
    val instances = extractInstances(dataset)

    val labels: Dataset[Double] = dataset.select(dataset($(labelCol)).as[Double](ExpressionEncoder()))
    val labelsSummary = labels.rdd.mapPartitions((data: Iterator[Double]) => {
      val result = data.foldLeft(new MultivariateOnlineSummarizer())(
        (summarizer, y) => summarizer.add(mllib.linalg.Vectors.dense(y)))
      Iterator(result)
    }).reduce(_ merge _)

    val yMean = labelsSummary.mean.asML(0)
    var yStd = Vectors.fromBreeze(breeze.numerics.sqrt(labelsSummary.variance.asBreeze))(0)
    yStd = if (yStd > 0) yStd else math.abs(yMean)

    val features: Dataset[Vector] = dataset.select(dataset($(featuresCol)).as[Vector](ExpressionEncoder()))
    val featuresSummary = features.rdd.mapPartitions((data: Iterator[Vector]) => {
      val result = data.foldLeft(new MultivariateOnlineSummarizer())(
        (summarizer, vector) => summarizer.add(mllib.linalg.Vectors.fromBreeze(vector.asBreeze)))
      Iterator(result)
    }).reduce(_ merge _)

    val featuresMean = featuresSummary.mean.toArray
    val featuresStd: Array[Double] = Vectors.fromBreeze(breeze.numerics.sqrt(featuresSummary.variance.asBreeze)).toArray

    var regularization: Option[DifferentiableRegularization[Vector]] = None
    if ($(regParam) != 0.0)
      regularization = Some(new L2Regularization($(regParam), _ => true, None))
    val costFun = new RDDLossFunction(
      instances,
//      b => new HuberAggregator(false, $(tol), bcFeaturesStd)(b),
      b => new LeastSquaresAggregator(
        yStd,
        yMean,
        true,
        instances.context.broadcast(featuresStd),
        instances.context.broadcast(featuresMean)
      )(b),
      regularization
    )
    val optimizer = new LBFGSB(
      BDenseVector.fill(numFeatures) {Double.MinValue},
      BDenseVector.fill(numFeatures) {Double.MaxValue},
      $(maxIter),
      10,
      $(tol)
    )

    val res = optimizer.iterations(costFun, BDenseVector.zeros(numFeatures))
      .toArray
      .last
    val coefficients = res.x
    for (i <- 0 until coefficients.size) {
      if (featuresStd(i) == 0.0)
        coefficients(i) = 0.0
      else
        coefficients(i) *= yStd / featuresStd(i)
    }
    val intercept = yMean - coefficients.dot(Vectors.dense(featuresMean).asBreeze)

    copyValues(new LinearRegressionModel(Vectors.fromBreeze(coefficients), intercept))
  }

  override def copy(extra: ParamMap): LinearRegression = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)
}

object LinearRegression extends DefaultParamsReadable[LinearRegression]

class LinearRegressionModel private[made](
                                           override val uid: String,
                                           val coefficients: DenseVector,
                                           val b: Double
                                         ) extends RegressionModel[Vector, LinearRegressionModel] with LinearRegressionParams with MLWritable {

  private[made] def this(coefficients: Vector, b: Double) =
    this(Identifiable.randomUID("linearRegressionModel"), coefficients.toDense, b)

  override def copy(extra: ParamMap): LinearRegressionModel = copyValues(
    new LinearRegressionModel(coefficients, b), extra
  )

  override def predict(features: Vector): Double = features.asBreeze.dot(coefficients.asBreeze) + b

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

  override def write: MLWriter = new DefaultParamsWriter(this) {
    override protected def saveImpl(path: String): Unit = {
      super.saveImpl(path)

      val vectors = coefficients.asInstanceOf[Vector] -> b.asInstanceOf[Double]

      sqlContext.createDataFrame(Seq(vectors)).write.parquet(path + "/vectors")
    }
  }
}

object LinearRegressionModel extends MLReadable[LinearRegressionModel] {
  override def read: MLReader[LinearRegressionModel] = new MLReader[LinearRegressionModel] {
    override def load(path: String): LinearRegressionModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc)

      val vectors = sqlContext.read.parquet(path + "/vectors")

      val (coefficients, b) =  vectors.select(
        vectors("_1").as[Vector](ExpressionEncoder[Vector]()),
        vectors("_2").as[Double](ExpressionEncoder[Double]())
      ).first()

      val model = new LinearRegressionModel(coefficients, b)
      metadata.getAndSetParams(model)
      model
    }
  }
}
