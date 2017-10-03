/**
  * Created by mxs15 on 5/3/2017.
  */
import org.apache.log4j.Logger
import org.apache.log4j.Level
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel
import org.apache.spark.mllib.recommendation.Rating
import org.apache.spark.rdd.RDD

object CoAuthor {
  def main(args: Array[String]): Unit = {
    //Create conf object
    val conf = new SparkConf().setAppName("Coauthor").setMaster("local")

    //create spark context object
    val sc = new SparkContext(conf)
    Logger.getLogger("all").setLevel(Level.OFF)
    val data = sc.textFile("F:\\Projecttttttttttttttt\\dblpCoA.txt")
    val data1 = data.map(s => s.trim.split(' '))
    val data2 = data1.map(a=>Array(a(0).toInt, a(1).toInt, 1.0))
    val ratings = data2.map(a=>Array(a(0), a(1), a(2)) match
    { case Array(user, item, rate) =>
        Rating(user.toInt, item.toInt, rate.toDouble)
    })

    val splits = ratings.randomSplit(Array(0.6, 0.2, 0.2))
    val training = splits(0)
    val validation = splits(1)
    val test = splits(2)

    // Training using ALS
    val ranks = List(10, 15, 20, 25, 30)
    val lambdas = List(0.01, 0.03, 0.05)
    val numIters = List(5, 10, 20)
    var bestValidationRmse = Double.MaxValue
    var bestRank = 0
    var bestLambda = -1.0
    var bestNumIter = -1
    var bestModel : MatrixFactorizationModel = null

    // Evaluate the model on validation data
    val usersProducts = validation.map {case Rating(user, product, rate) =>(user, product)}

    // choose the best rank, lambda and number of iterations
    for (rank <- ranks; lambda <- lambdas; numIter <- numIters) {
      val model = ALS.train(training, rank, numIter, lambda)

      val predictions = model.predict(usersProducts).map {
        case Rating(user, product, rate) =>((user, product), rate)}

      val ratesAndPreds = validation.map {
        case Rating(user, product, rate) =>((user, product), rate)}.join(predictions)

      val MSE = ratesAndPreds.map { case ((user, product), (r1, r2)) =>
        val err = (r1 - r2)
        err * err
      }.mean()

      val validationRmse = MSE
      println("RMSE (validation) = " + validationRmse + " for the model trained with rank = "
        + rank + ", lambda = " + lambda + ", and numIter = " + numIter + ".")
      if (validationRmse < bestValidationRmse) {
        bestValidationRmse = validationRmse
        bestRank = rank
        bestLambda = lambda
        bestNumIter = numIter
        bestModel = model
      }
    }

    // Obtain the best model
    //val bestModel = ALS.train(training, bestRank, bestNumIter, bestLambda)
    // Evaluate model on test data
    val usersProducts1 = test.map {
      case Rating(user, product, rate) =>(user, product)}
    val predictions1 = bestModel.predict(usersProducts1).map {
      case Rating(user, product, rate) =>((user, product), rate)}
    val ratesAndPreds1 = test.map {
      case Rating(user, product, rate) =>((user, product), rate)}.join(predictions1)
    val MSE1 = ratesAndPreds1.map { case ((user, product), (r1, r2)) =>
      val err = (r1 - r2)
      err * err
    }.mean()

    val testRmse = MSE1
    println("The best model was trained with rank = " + bestRank + " and lambda = " + bestLambda
      + ", and numIter = " + bestNumIter + ", and its RMSE on the test set is " + MSE1 + ".")


    // recommend authors to collaborate with for a given author id 2
    val coauthors = ratings.map { case Rating(user, product, rate) =>
      (2, product)
    }
    val final_predictions =
      bestModel.predict(coauthors).map { case Rating(user, product, rate) =>
        (user, product, rate)
      }

    println("Authors recommended for author with id 2, format(authorid, recommend authorid, likelihood):")
    val result = final_predictions.distinct.top(50){Ordering.by(x => x._3)}.foreach(println)
    println(bestRank+ "  "+ bestLambda + " "+  bestNumIter + " " + MSE1)

    //bestModel.save(sc, "BestModelForCoAuthor")

    sc.stop()
  }
}
