import optuna



def catboost_space(trial, config):

    params = config.tuning.models.catboost


    return {


        "iterations":

            trial.suggest_int(

                "iterations",

                params.iterations.min,

                params.iterations.max

            ),



        "depth":

            trial.suggest_int(

                "depth",

                params.depth.min,

                params.depth.max

            ),



        "learning_rate":

            trial.suggest_float(

                "learning_rate",

                params.learning_rate.min,

                params.learning_rate.max,

                log=params.learning_rate.log

            ),



        "l2_leaf_reg":

            trial.suggest_float(

                "l2_leaf_reg",

                params.l2_leaf_reg.min,

                params.l2_leaf_reg.max

            )

    }



def xgboost_space(trial, config):

    params = config.tuning.models.xgboost


    return {


        "n_estimators":

            trial.suggest_int(

                "n_estimators",

                params.n_estimators.min,

                params.n_estimators.max

            ),



        "max_depth":

            trial.suggest_int(

                "max_depth",

                params.max_depth.min,

                params.max_depth.max

            ),



        "learning_rate":

            trial.suggest_float(

                "learning_rate",

                params.learning_rate.min,

                params.learning_rate.max,

                log=params.learning_rate.log

            ),



        "subsample":

            trial.suggest_float(

                "subsample",

                params.subsample.min,

                params.subsample.max

            ),



        "colsample_bytree":

            trial.suggest_float(

                "colsample_bytree",

                params.colsample_bytree.min,

                params.colsample_bytree.max

            ),



        "min_child_weight":

            trial.suggest_int(

                "min_child_weight",

                params.min_child_weight.min,

                params.min_child_weight.max

            )

    }