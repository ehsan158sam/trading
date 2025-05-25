# D:\AdvancedTradingSystem\modules\insight_dashboard.py
# Version: Added 'from pathlib import Path' and fixed use_reloader

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from pathlib import Path # <--- *** این خط اضافه شد ***

import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np 

try:
    from config.schemas import InsightDashboardSettings
except ImportError:
    print("WARNING (insight_dashboard.py): Could not import schemas. Using placeholder.")
    class InsightDashboardSettings: enabled=False; host="127.0.0.1"; port=8050; debug_mode=True; update_interval_seconds=15

logger = logging.getLogger(__name__)

class DashboardDataProvider:
    def __init__(self, orchestrator_ref: Optional[Any] = None):
        self.orchestrator = orchestrator_ref
        self.last_update_time = datetime.now()

    async def get_system_status(self) -> Dict[str, Any]:
        teg_connected = False
        ai_status = "Unknown"
        if self.orchestrator:
            if hasattr(self.orchestrator, 'trade_execution_gateway') and self.orchestrator.trade_execution_gateway:
                teg_connected = getattr(self.orchestrator.trade_execution_gateway, '_is_connected', False)
            if hasattr(self.orchestrator, 'synapse_ai_core') and self.orchestrator.synapse_ai_core:
                ai_status = "Idle/Trading" if any(self.orchestrator.synapse_ai_core.models.values()) else "No Models"
                if hasattr(self.orchestrator.synapse_ai_core, '_active_training_tasks') and self.orchestrator.synapse_ai_core._active_training_tasks:
                    ai_status = "Training"
        return {
            "system_running": True,
            "broker_connection": "Connected" if teg_connected else "Disconnected",
            "ai_status": ai_status
        }

    async def get_account_summary(self) -> Dict[str, Any]:
        if self.orchestrator and hasattr(self.orchestrator, 'trade_execution_gateway') and self.orchestrator.trade_execution_gateway:
            acc_info = await self.orchestrator.trade_execution_gateway.get_account_info()
            if acc_info:
                return {
                    "balance": acc_info.get('balance', 0), "equity": acc_info.get('equity', 0),
                    "profit": acc_info.get('profit', 0), "margin_free": acc_info.get('margin_free', 0),
                    "currency": acc_info.get('currency', 'USD')
                }
        return {"balance": 0, "equity": 0, "profit": 0, "margin_free": 0, "currency": "N/A"}

    async def get_equity_curve_data(self) -> pd.DataFrame:
        num_points = 30
        dates = pd.to_datetime([datetime.now() - timedelta(days=i) for i in range(num_points, 0, -1)])
        initial_balance = 10000.0
        if self.orchestrator and self.orchestrator.settings and hasattr(self.orchestrator.settings, 'synapse_ai') and self.orchestrator.settings.synapse_ai:
            initial_balance = self.orchestrator.settings.synapse_ai.env_initial_balance
        equity = np.cumsum(np.random.randn(num_points) * 100) + initial_balance
        return pd.DataFrame({"time": dates, "equity": equity})

    async def get_open_positions(self) -> List[Dict[str, Any]]:
        if self.orchestrator and hasattr(self.orchestrator, 'trade_execution_gateway') and self.orchestrator.trade_execution_gateway:
            open_pos_raw = await self.orchestrator.trade_execution_gateway.get_open_positions()
            return open_pos_raw or []
        return []

    async def get_trade_history(self, limit=50) -> pd.DataFrame:
        if self.orchestrator and hasattr(self.orchestrator, 'trade_execution_gateway') and self.orchestrator.trade_execution_gateway:
             deals_raw = await self.orchestrator.trade_execution_gateway.get_historical_trades(
                 from_date=datetime.now() - timedelta(days=7),
                 to_date=datetime.now()
             )
             if deals_raw:
                 df_deals = pd.DataFrame(deals_raw)
                 cols_map = {'time_msc': 'time', 'symbol': 'symbol', 'type': 'deal_type', 'entry': 'deal_entry', 
                             'volume': 'volume', 'price': 'price', 'profit': 'pnl', 'commission': 'commission', 'comment': 'comment'}
                 # اطمینان از اینکه فقط ستون های موجود انتخاب می شوند
                 existing_cols_in_df = [k_mt5 for k_mt5 in cols_map.keys() if k_mt5 in df_deals.columns]
                 if not existing_cols_in_df: # اگر هیچکدام از ستون های مورد انتظار نیستند
                     logger.warning(f"Trade history from TEG does not contain expected columns. Available: {df_deals.columns.tolist()}")
                     return pd.DataFrame()
                 
                 df_deals_show = df_deals[existing_cols_in_df].rename(columns=cols_map)
                 
                 if 'time' in df_deals_show:
                     df_deals_show['time'] = pd.to_datetime(df_deals_show['time'], unit='ms', utc=True).dt.tz_localize(None)
                 return df_deals_show.sort_values(by='time', ascending=False).head(limit)
        return pd.DataFrame()

    async def get_system_logs(self, level: str = "INFO", limit: int = 20) -> List[str]:
        log_file_path_obj = None
        # اطمینان از اینکه orchestrator و تنظیمات آن قبل از دسترسی به log_file_path وجود دارند
        if (self.orchestrator and 
            hasattr(self.orchestrator, 'settings') and self.orchestrator.settings and
            hasattr(self.orchestrator.settings, 'logging') and self.orchestrator.settings.logging and
            self.orchestrator.settings.logging.log_file_path):
            
            log_file_path_str = self.orchestrator.settings.logging.log_file_path
            log_file_path_obj = Path(log_file_path_str) # Path اینجا استفاده می شود
            if not log_file_path_obj.is_absolute() and hasattr(self.orchestrator, 'config_path') and self.orchestrator.config_path.is_file():
                log_file_path_obj = (self.orchestrator.config_path.parent / log_file_path_obj).resolve()

        if log_file_path_obj and log_file_path_obj.exists():
            try:
                with open(log_file_path_obj, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                return [line.strip() for line in lines[-limit:]]
            except Exception as e:
                logger.error(f"Error reading log file {log_file_path_obj}: {e}")
                return [f"Error reading log file: {e}"]
        logger.debug(f"Log file not found or not configured for dashboard. Path checked: {log_file_path_obj}")
        return [f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {level} - Sample log (log file not found or not configured)."]


class InsightDashboard:
    def __init__(self, settings: InsightDashboardSettings, data_provider: DashboardDataProvider):
        self.settings = settings
        self.data_provider = data_provider
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG, dbc.icons.FONT_AWESOME])
        self._setup_layout()
        self._setup_callbacks()

    def _setup_layout(self):
        self.app.layout = dbc.Container(fluid=True, children=[
            dcc.Interval(id='interval-component', interval=self.settings.update_interval_seconds * 1000, n_intervals=0),
            dbc.Row(dbc.Col(html.H2(children=[html.I(className="fas fa-chart-line me-2"),"Advanced Trading System Dashboard"], className="text-center text-primary mb-4 pt-3"), width=12)),
            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody(id='system-status-card')), lg=4, md=6, className="mb-3"),
                dbc.Col(dbc.Card(dbc.CardBody(id='account-summary-card')), lg=8, md=6, className="mb-3"),
            ]),
            dbc.Row(dbc.Col(dbc.Card([dbc.CardHeader("Equity Over Time"), dbc.CardBody(dcc.Graph(id='equity-curve-graph'))]), width=12, className="mb-3")),
            dbc.Row([
                dbc.Col(dbc.Card([dbc.CardHeader("Open Positions"), dbc.CardBody(id='open-positions-table')]), lg=7, className="mb-3"),
                dbc.Col(dbc.Card([dbc.CardHeader("Recent Trade History"), dbc.CardBody(id='trade-history-table')]), lg=5, className="mb-3"),
            ]),
            dbc.Row(dbc.Col(dbc.Card([dbc.CardHeader("System Logs"), dbc.CardBody(html.Pre(id='system-logs-output', style={'maxHeight': '250px', 'overflowY': 'scroll', 'fontSize': '0.8em'}))]), width=12, className="mb-3")),
            html.Footer(dbc.Row(dbc.Col(html.P(id="footer-timestamp", className="text-center text-muted small"), width=12)))
        ])

    def _setup_callbacks(self):
        @self.app.callback(
            [Output('system-status-card', 'children'), Output('account-summary-card', 'children'),
             Output('equity-curve-graph', 'figure'), Output('open-positions-table', 'children'),
             Output('trade-history-table', 'children'), Output('system-logs-output', 'children'),
             Output('footer-timestamp', 'children')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_dashboard_data_callback(n_intervals):
            try:
                status_data = asyncio.run(self.data_provider.get_system_status())
                acc_summary = asyncio.run(self.data_provider.get_account_summary())
                equity_df = asyncio.run(self.data_provider.get_equity_curve_data())
                open_positions_data = asyncio.run(self.data_provider.get_open_positions())
                trade_history_df = asyncio.run(self.data_provider.get_trade_history(limit=10))
                logs_data = asyncio.run(self.data_provider.get_system_logs(limit=15))
            except Exception as e_callback_async:
                logger.error(f"Error running async data provider methods in dashboard callback: {e_callback_async}", exc_info=True)
                # برگرداندن مقادیر پیش فرض یا پیام خطا برای جلوگیری از شکست کامل داشبورد
                error_message = f"Error updating data: {e_callback_async}"
                return ([html.P(error_message)]*6) + [datetime.now().strftime('%Y-%m-%d %H:%M:%S')]


            status_card_content = [
                html.H5(children=[html.I(className="fas fa-cogs me-1"), "System Status"], className="card-title"),
                html.P(f"Broker: {status_data.get('broker_connection', 'N/A')}"),
                html.P(f"AI Status: {status_data.get('ai_status', 'N/A')}"),
            ]
            summary_card_content = [
                html.H5(children=[html.I(className="fas fa-wallet me-1"), "Account Summary"], className="card-title"),
                dbc.Row([
                    dbc.Col(f"Balance: {acc_summary.get('balance', 0):.2f} {acc_summary.get('currency', '')}", width=6),
                    dbc.Col(f"Equity: {acc_summary.get('equity', 0):.2f} {acc_summary.get('currency', '')}", width=6),
                ]),
                dbc.Row([
                    dbc.Col(f"Profit: {acc_summary.get('profit', 0):.2f} {acc_summary.get('currency', '')}", width=6),
                    dbc.Col(f"Free Margin: {acc_summary.get('margin_free', 0):.2f} {acc_summary.get('currency', '')}", width=6),
                ])
            ]
            equity_fig = go.Figure()
            if not equity_df.empty and 'time' in equity_df.columns and 'equity' in equity_df.columns:
                 equity_fig.add_trace(go.Scatter(x=equity_df['time'], y=equity_df['equity'], mode='lines+markers', name='Equity'))
            equity_fig.update_layout(template="plotly_dark", margin=dict(l=10, r=10, t=30, b=10), height=300)

            open_pos_table_content = html.P("No open positions.")
            if open_positions_data:
                open_pos_df = pd.DataFrame(open_positions_data)
                if not open_pos_df.empty:
                    cols_to_show_open = ['ticket', 'symbol', 'type_str', 'volume', 'price_open', 'sl', 'tp', 'profit']
                    # فقط ستون هایی که واقعا در DataFrame هستند را انتخاب کن
                    open_pos_df_show = open_pos_df[[col for col in cols_to_show_open if col in open_pos_df.columns]]
                    if not open_pos_df_show.empty:
                         open_pos_table_content = dbc.Table.from_dataframe(open_pos_df_show, striped=True, bordered=True, hover=True, responsive=True, size="sm", dark=True)
            
            trade_hist_table_content = html.P("No recent trade history.")
            if not trade_history_df.empty:
                trade_hist_table_content = dash_table.DataTable(
                    data=trade_history_df.to_dict('records'),
                    columns=[{'name': i.replace('_',' ').title(), 'id': i} for i in trade_history_df.columns],
                    style_cell={'textAlign': 'left', 'backgroundColor': '#2B3035', 'color': 'white', 'border': '1px solid #444'},
                    style_header={'backgroundColor': '#1E2125', 'fontWeight': 'bold', 'borderBottom': '2px solid #555'},
                    page_size=5, sort_action="native", style_as_list_view=True,
                    style_table={'overflowX': 'auto'}
                )
            
            logs_output_content = "\n".join(logs_data) if logs_data else "No logs to display."
            footer_text_content = f"Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

            return (status_card_content, summary_card_content, equity_fig, 
                    open_pos_table_content, trade_hist_table_content, logs_output_content, footer_text_content)

    def run(self):
        if not self.settings.enabled:
            logger.info("InsightDashboard is disabled in settings.")
            return
        
        logger.info(f"Starting InsightDashboard server on http://{self.settings.host}:{self.settings.port}")
        try:
            self.app.run(
                host=self.settings.host,
                port=self.settings.port,
                debug=self.settings.debug_mode,
                use_reloader=False # <--- این خط برای جلوگیری از خطای signal handler مهم است
            )
        except Exception as e:
            logger.error(f"Failed to start InsightDashboard server: {e}", exc_info=True)

    async def shutdown(self):
        logger.info("InsightDashboard shutting down...")
        # برای سرور Dash که در ترد جداگانه اجرا می شود، متوقف کردن آن از اینجا به طور مستقیم دشوار است
        # معمولا با خاموش شدن برنامه اصلی، ترد آن هم بسته می شود.
        # یا اگر از سرور پروداکشن استفاده شود، آن سرور جداگانه متوقف می شود.
        logger.info("InsightDashboard shutdown process initiated (actual server stop depends on thread/process management).")