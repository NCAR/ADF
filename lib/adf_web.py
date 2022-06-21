"""
Website (web) generation class for the
Atmospheric Diagnostics Framework (ADF).
This class inherits from the AdfObs class.

Currently this class does three things:

1.  Initializes an instance of AdfObs.

2.  Determines if a website will be generated.

3.  Sets website-related internal ADF variables.

This class also provides a method for generating
a website, as well as a method to add an image
file or pandas dataframe to the website.
"""

#++++++++++++++++++++++++++++++
#Import standard python modules
#++++++++++++++++++++++++++++++

import os

from pathlib import Path

#+++++++++++++++++++++++++++++++++++++++++++++++++
#import non-standard python modules, including ADF
#+++++++++++++++++++++++++++++++++++++++++++++++++

#ADF modules:
from adf_obs import AdfObs

#+++++++++++++++++++
#Define Obs class
#+++++++++++++++++++

class AdfWeb(AdfObs):

    """
    Website class, which initializes
    an AdfObs object and provides
    additional variables and methods
    needed for website generation.
    """

    def __init__(self, config_file, debug=False):

        """
        Initalize ADF Web object.
        """

        #Initialize Obs attributes:
        super().__init__(config_file, debug=debug)

    #########

    # Create property needed to return "create_html" logical to user:
    @property
    def create_html(self):
        """Return the "create_html" logical to user if requested."""
        return self.get_basic_info('create_html')

    #########

    def create_website(self):

        """
        Generate webpages to display diagnostic results.
        """

        #import needed standard modules:
        import shutil
        import itertools
        from collections import OrderedDict

        #Import "special" modules:
        try:
            import jinja2
        except ImportError:
            emsg = "Jinja2 module does not exist in python path, but is needed for website."
            emsg += "\nPlease install module, e.g. 'pip install Jinja2'"
            self.end_diag_fail(emsg)
        #End except

        #Notify user that script has started:
        print("\n  Generating Diagnostics webpages...")

        #Check where the relevant plots are located:
        if self.plot_location:
            plot_location = self.plot_location
        else:
            self._append_plot_loc(self.get_basic_info('cam_diag_plot_loc', required=True))
        #End if

        #If there is more than one plot location, then create new website directory:
        if len(plot_location) > 1:
            main_site_path = Path(self.get_basic_info('cam_diag_plot_loc', required=True))
            main_site_path = main_site_path / "main_website"
            main_site_path.mkdir(exist_ok=True)
            case_sites = OrderedDict()
        else:
            main_site_path = "" #Set main_site_path to blank value
        #End if

        #Extract needed variables from yaml file:
        case_names = self.get_cam_info('cam_case_name', required=True)

        #Extract variable list:
        var_list = self.diag_var_list

        #Set name of comparison data, which depends on "compare_obs":
        if self.compare_obs:
            data_name = "obs"
        else:
            data_name = self.get_baseline_info('cam_case_name', required=True)
        #End if

        #Set preferred order of seasons:
        season_order = ["ANN", "DJF", "MAM", "JJA", "SON"]

        # Variable categories
        var_cat_dict = {
            'Clouds': {'ACTNI', 'ACTNL', 'ACTREI', 'ACTREL', 'ADRAIN', 'ADSNOW',
                       'AREI', 'AREL', 'CCN3', 'CDNUMC', 'CLDHGH', 'CLDICE',
                       'CLDLIQ', 'CLDLOW', 'CLDMED', 'CLDTOT', 'CLOUD', 'CONCLD',
                       'EVAPPREC', 'EVAPSNOW', 'FCTI', 'FCTL', 'FICE', 'FREQI',
                       'FREQL', 'FREQR', 'FREQS', 'MPDQ', 'PRECC', 'PRECL',
                       'PRECSC', 'PRECSL', 'PRECT', 'TGCLDIWP', 'TGCLDLWP'},
            'Deep Convection': {'CAPE', 'CMFMC_DP', 'FREQZM', 'ZMDQ', 'ZMDT'},
            'COSP': {'CLDTOT_ISCCP', 'CLIMODIS', 'CLTMODIS', 'CLWMODIS',
                     'FISCCP1_COSP', 'ICE_ICLD_VISTAU', 'IWPMODIS',
                     'LIQ_ICLD_VISTAU', 'LWPMODIS', 'MEANCLDALB_ISCCP',
                     'MEANPTOP_ISCCP', 'MEANTAU_ISCCP', 'MEANTB_ISCCP',
                     'MEANTBCLR_ISCCP', 'PCTMODIS', 'REFFCLIMODIS', 'REFFCLWMODIS',
                     'SNOW_ICLD_VISTAU', 'TAUTMODIS', 'TAUWMODIS',
                     'TOT_CLD_VISTAU', 'TOT_ICLD_VISTAU'},
            'Budget': {'DCQ', 'DQCORE', 'DTCORE', 'MPDICE', 'MPDLIQ', 'PTEQ'},
            'Radiation': {'FLNS', 'FLNSC', 'FLNT', 'FLNTC', 'FLUT', 'FSDS',
                          'FSDSC', 'FSNS', 'FSNSC', 'FSNT', 'FSNTC', 'FSNTOA',
                          'LHFLX', 'LWCF', 'QRL', 'QRS', 'SHFLX', 'SWCF'},
            'State': {'OMEGA', 'OMEGA500', 'PINT', 'PMID', 'PS', 'PSL', 'Q',
                      'RELHUM', 'T', 'U', 'V', 'Z3', 'Wind'},
            'Surface': {'PBLH', 'QFLX', 'TAUX', 'TAUY', 'TREFHT', 'U10',
                        'Surface_Wind_Stress'},
            'GW': {'QTGW', 'UGTW_TOTAL', 'UTGWORO', 'VGTW_TOTAL', 'VTGWORO'},
            'CLUBB': {'RVMTEND_CLUBB', 'STEND_CLUBB', 'WPRTP_CLUBB', 'WPTHLP_CLUBB'}
        }

        #Set preferred order of plot types:
        #Make dictionaries for both html paths and plot type names for website
        #NOTE there may be a better way to do this with an Ordered Dict, but the
        #polar plot having more than one plot made it tricky.
        ptype_html_dict = {'global_latlon_map': ['html_img/mean_diag_LatLon.html'],
                           'zonal_mean': ['html_img/mean_diag_Zonal.html'],
                           'global_latlon_vect_map': ['html_img/mean_diag_LatLon_Vector.html'],
                           'polar_map': ['html_img/mean_diag_NHPolar.html',
                                         'html_img/mean_diag_SHPolar.html'],
                            'cam_taylor_diagram': ["html_img/mean_diag_TaylorDiag.html"]}

        ptype_order_dict = {'global_latlon_map': ["LatLon"],
                            'zonal_mean': ["Zonal"],
                            'global_latlon_vect_map': ["LatLon_Vector"],
                            'polar_map': ["NHPolar","SHPolar"],
                            'cam_taylor_diagram': ["TaylorDiag"]}

        #Grab the plot type functions form user
        plot_func_names = self.read_config_var('plotting_scripts')

        #Since polar has more than one plot type name, make a list of lists
        #that grab all the paths and names
        ptype_html = sorted([ptype_html_dict[x] for x in plot_func_names if x in ptype_html_dict])
        ptype_order = sorted([ptype_order_dict[x] for x in plot_func_names if x in ptype_order_dict])

        #Flatten the list of lists into a regular list
        ptype_html_list = list(itertools.chain.from_iterable(ptype_html))
        ptype_order_list = list(itertools.chain.from_iterable(ptype_order))

        #Make dictionary for plot type names and html paths
        plot_type_html = dict(zip(ptype_order_list, ptype_html_list))

        main_title = "CAM Diagnostics"

        #Check if any variables are associated with specific vector quantities,
        #and if so then add the vectors to the website variable list.
        for var in var_list:
            if var in self.variable_defaults:
                vect_name = self.variable_defaults[var].get("vector_name", None)
                if vect_name and (vect_name not in var_list):
                    var_list.append(vect_name)
                #End if
            #End if
        #End for

        #Extract pressure levels being plotted:
        pres_levs = self.get_basic_info("plot_press_levels")

        if pres_levs:
            #Create pressure-level variable dictionary:
            pres_levs_var_dict = {}

            #Now add variables on pressure levels, if applicable.
            #Please note that this method is not particularly
            #efficient as most of these variables won't actually exist:
            for var in var_list:
                #Find variable category:
                category = next((cat for cat, varz in var_cat_dict.items() if var in varz), None)

                #Add variable with pressure levels:
                #Please note that this method is not particularly
                #efficient as most of these variables won't actually exist:
                for pres in pres_levs:
                    if category:
                        if category in pres_levs_var_dict:
                            pres_levs_var_dict[category].append(f"{var}_{pres}hpa")
                        else:
                            pres_levs_var_dict[category] = [f"{var}_{pres}hpa"]
                        #End if
                    else:
                        if "none" in pres_levs_var_dict:
                            pres_levs_var_dict["none"].append(f"{var}_{pres}hpa")
                        else:
                            pres_levs_var_dict["none"] = [f"{var}_{pres}hpa"]
                        #End if
                    #End if
                #End for
            #End for

            #Now loop over pressure variable dictionary:
            for category, pres_var_names in pres_levs_var_dict.items():
                #Add pressure-level variable to category if applicable:
                if category in var_cat_dict:
                    var_cat_dict[category].update(pres_var_names)
                #End if

                #Add pressure-level variable to variable list:
                var_list.extend(pres_var_names)

            #End for
        #End if

        # add fake "cam" variable to variable list in order to find Taylor diagram plots:
        var_list.append('cam')

        #Determine local directory:
        adf_lib_dir = Path(__file__).parent

        #Set path to Jinja2 template files:
        jinja_template_dir = Path(adf_lib_dir, 'website_templates')

        #Create the jinja Environment object:
        jinenv = jinja2.Environment(loader=jinja2.FileSystemLoader(jinja_template_dir))

        #Create alphabetically-sorted variable list:
        var_list_alpha = sorted(var_list)

        #Loop over model cases:
        for case_idx, case_name in enumerate(case_names):

            #Create new path object from user-specified plot directory path:
            plot_path = Path(plot_location[case_idx])

            #Create the directory where the website will be built:
            website_dir = plot_path / "website"
            website_dir.mkdir(exist_ok=True)

            #Create a directory that will hold just the html files for individual images:
            img_pages_dir = website_dir / "html_img"
            img_pages_dir.mkdir(exist_ok=True)

            #Create a directory that will hold copies of the actual images:
            assets_dir = website_dir / "assets"
            assets_dir.mkdir(exist_ok=True)

            #Specify where CSS files will be stored:
            css_files_dir = website_dir / "templates"
            css_files_dir.mkdir(exist_ok=True)

            #Copy CSS files over to output directory:
            for css_file in jinja_template_dir.glob('*.css'):
                shutil.copyfile(css_file, css_files_dir / css_file.name)
            #End for

            #Copy images into the website image dictionary:
            for img in plot_path.glob("*.png"):
                idest = assets_dir / img.name
                shutil.copyfile(img, idest) # store image in assets
            #End for

            #Loop over plot type:
            for ptype in ptype_order_list:
                # this is going to hold the data for building the mean
                # plots provisional structure:
                # key = variable_name
                # values -> dict w/ keys being "TYPE" of plots
                # w/ values being dict w/ keys being TEMPORAL sampling,
                # values being the URL
                mean_html_info = OrderedDict()

                for var in var_list_alpha:
                    #Loop over seasons:
                    for season in season_order:

                        #Create the data that will be fed into the template:
                        for img in assets_dir.glob(f"{var}_{season}_{ptype}_Mean*.png"):

                            #Create output file (don't worry about analysis type for now):
                            outputfile = img_pages_dir / f'plot_page_{var}_{season}_{ptype}.html'

                            # Search through all categories and see
                            # which one the current variable is part of
                            category = next((cat for cat, varz \
                                             in var_cat_dict.items() if var in varz), None)
                            if not category:
                                category = 'No category yet'
                            #End if

                            if category not in mean_html_info:
                                mean_html_info[category] = OrderedDict()

                            #Initialize Ordered Dictionary for variable:
                            if var not in mean_html_info[category]:
                                mean_html_info[category][var] = OrderedDict()

                            #Initialize Ordered Dictionary for plot type:
                            if ptype not in mean_html_info[category][var]:
                                mean_html_info[category][var][ptype] = OrderedDict()

                            #Initialize Ordered Dictionary for season:
                            if season not in mean_html_info[category][var][ptype]:
                                mean_html_info[category][var][ptype][season] = OrderedDict()

                            mean_html_info[category][var][ptype][season] = outputfile.name

                #Loop over variables:
                for var in var_list_alpha:
                    #Loop over seasons:
                    for season in season_order:
                        #Create the data that will be fed into the template:
                        for img in assets_dir.glob(f"{var}_{season}_{ptype}_Mean*.png"):
                            alt_text  = img.stem #Extract image file name text

                            #Create output file (don't worry about analysis type for now):
                            outputfile = img_pages_dir / f'plot_page_{var}_{season}_{ptype}.html'
                            # Hacky - how to get the relative path in a better way?:
                            img_data = [os.pardir+os.sep+assets_dir.name+os.sep+img.name, alt_text]

                            #Create titles
                            var_title = f"Variable: {var}"
                            season_title = f"Season: {season}"
                            plottype_title = f"Plot: {ptype}"
                            tmpl = jinenv.get_template('template.html')  #Set template
                            rndr = tmpl.render(title=main_title,
                                               var_title=var_title,
                                               season_title=season_title,
                                               plottype_title=plottype_title,
                                               imgs=img_data,
                                               case1=case_name,
                                               case2=data_name,
                                               mydata=mean_html_info,
                                               plot_types=plot_type_html) #The template rendered

                            #Open HTML file:
                            with open(outputfile, 'w', encoding='utf-8') as ofil:
                                ofil.write(rndr)
                            #End with

                            #Construct individual plot type mean_diag html files
                            mean_tmpl = jinenv.get_template(f'template_mean_diag_{ptype}.html')
                            mean_rndr = mean_tmpl.render(title=main_title,
                                            case1=case_name,
                                            case2=data_name,
                                            mydata=mean_html_info,
                                            plot_types=plot_type_html)

                            #Write mean diagnostic plots HTML file:
                            outputfile = img_pages_dir / f"mean_diag_{ptype}.html"
                            with open(outputfile,'w', encoding='utf-8') as ofil:
                                ofil.write(mean_rndr)
                            #End with
                        #End for (assests loop)
                    #End for (seasons loop)

            #Grab AMWG Table HTML files:
            table_html_files = list(plot_path.glob(f"amwg_table_{case_name}*.html"))

            #Grab the comparison table and move it to website dir
            comp_table_html_file = list(plot_path.glob("*comp.html"))

            #Also grab baseline/obs tables, which are always stored in the first case directory:
            if case_idx == 0:
                data_table_html_files = list(plot_path.glob(f"amwg_table_{data_name}*.html"))
            #End if

            #Determine if any AMWG tables were generated:
            if table_html_files:

                #Set Table HTML generation logical to "TRUE":
                gen_table_html = True

                #Create a directory that will hold table html files:
                table_pages_dir = website_dir / "html_table"
                table_pages_dir.mkdir(exist_ok=True)

                #Move all case table html files to new directory:
                for table_html in table_html_files:
                    shutil.move(table_html, table_pages_dir / table_html.name)
                #End for

                #copy all data table html files as well:
                for data_table_html in data_table_html_files:
                    shutil.copy2(data_table_html, table_pages_dir / data_table_html.name)
                #End for

                #Construct dictionary needed for HTML page:
                amwg_tables = OrderedDict()

                for case in [case_name, data_name]:

                    #Search for case name in moved HTML files:
                    table_htmls = sorted(table_pages_dir.glob(f"amwg_table_{case}.html"))

                    #Check if file exists:
                    if table_htmls:

                        #Initialize loop counter:
                        count = 0

                        #Loop over globbed files:
                        for table_html in table_htmls:

                            #Create relative path for HTML file:
                            amwg_tables[case] = table_html.name

                            #Update counter:
                            count += 1

                            #If counter greater than one, then throw an error:
                            if count > 1:
                                emsg = f"More than one AMWG table is associated with case '{case}'."
                                emsg += "\nNot sure what is going on, "
                                emsg += "\nso website generation will end here."
                                self.end_diag_fail(emsg)
                            #End if
                        #End for (table html file loop)
                    #End if (table html file exists check)
                #End for (case vs data)

                #Check if comp table exists
                #(if not, then obs are being compared and comp table is not created)
                if comp_table_html_file:
                    #Move the comparison table html file to new directory
                    for comp_table in comp_table_html_file:
                        shutil.move(comp_table, table_pages_dir / comp_table.name)
                        #Add comparison table to website dictionary
                        # * * * * * * * * * * * * * * * * * * * * * * * * * * * *
                        #This will be for single-case for now,
                        #will need to think how to change as multi-case is introduced
                        amwg_tables["Case Comparison"] = comp_table.name
                        # * * * * * * * * * * * * * * * * * * * * * * * * * * * *
                    #End for

                # need this to grab the locations of the amwg tables...
                amwg_table_data = [str(table_pages_dir / table_html.name), ""]

                #Construct mean_table.html
                mean_tmpl = jinenv.get_template('template_mean_table.html')
                mean_rndr = mean_tmpl.render(title=main_title,
                                value=amwg_table_data,
                                case1=case_name,
                                case2=data_name,
                                amwg_tables=amwg_tables,
                                plot_types=plot_type_html,
                                )

                #Write mean diagnostic tables HTML file:
                outputfile = table_pages_dir / "mean_table.html"
                with open(outputfile, 'w', encoding='utf-8') as ofil:
                    ofil.write(mean_rndr)
                #End with
            else:
                #No Tables exist, so no link will be added to main page:
                gen_table_html = False
            #End if

            #Construct index.html
            #index_title = "AMP Diagnostics Prototype"
            index_tmpl = jinenv.get_template('template_index.html')
            index_rndr = index_tmpl.render(title=main_title,
                             case1=case_name,
                             case2=data_name,
                             gen_table_html=gen_table_html,
                             plot_types=plot_type_html,
                             )

            #Write Mean diagnostics HTML file:
            outputfile = website_dir / "index.html"
            with open(outputfile, 'w', encoding='utf-8') as ofil:
                ofil.write(index_rndr)
            #End with

            #If this is a multi-case instance, then copy website to "main" directory:
            if main_site_path:
                shutil.copytree(website_dir, main_site_path / case_name)
                #Also add path to case_sites dictionary:
                case_sites[case_name] = os.path.join(os.curdir, case_name, "index.html")
                #Finally, if first case, then also copy templates directory for CSS files:
                if case_idx == 0:
                    shutil.copytree(css_files_dir, main_site_path / "templates")
                #End if
            #End if
        #End for (model case loop)

        #Create multi-case site, if needed:
        if main_site_path:
            main_title = "ADF Diagnostics"
            main_tmpl = jinenv.get_template('template_multi_case_index.html')
            main_rndr = main_tmpl.render(title=main_title,
                            case_sites=case_sites,
                            )
            #Write multi-case main HTML file:
            outputfile = main_site_path / "index.html"
            with open(outputfile, 'w', encoding='utf-8') as ofil:
                ofil.write(main_rndr)
            #End with
        #End if

        #Notify user that script has finishedd:
        print("  ...Webpages have been generated successfully.")

#++++++++++++++++++++
#End Class definition
#++++++++++++++++++++
